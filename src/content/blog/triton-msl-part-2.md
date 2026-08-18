---
title: "An Update to triton-msl: Part-1 Clarifications, Attention That Beats PyTorch, Quantized Inference, and AMD"
description: "Part 2 of the Metal backend for Triton. A dispatch fix makes FlashAttention beat PyTorch and hold its own against Apple's MLX, weight-only int8/int4 decode hits the memory roofline, byte-identical output extends to a third vendor (AMD), and macOS 26 cost zero codegen changes."
pubDate: 2026-08-18T12:00:00Z
heroImage: '../../assets/triton-msl-part-2-hero.png'
---

This is a follow-up to the first post on triton-msl, a Metal backend for Triton. The point of it is to keep kernel work on the Mac you already own. You write and debug there, and when you need datacenter silicon, you run the same `@triton.jit` source unchanged. Only the backend lowering differs, so the output comes out byte-identical across Apple and NVIDIA. No GPU rental just to iterate, and no rewrite to deploy. Part 1 proved that concept. This update takes it further in the two places that were still open: how fast the kernels run on Apple Silicon itself, and how far the byte-identical guarantee reaches. It also picks up the design questions readers raised afterward. Most of the work went into performance, and attention is where it shows most: part 1 left it behind hand-tuned Metal and MLX in absolute speed, and it is now the part I would point at first.

All of the numbers below come from an M4 Max on macOS 26.6 with torch 2.12, measured cold with alternating A/B runs to cancel thermal drift. Every comparison against MLX uses Apple's own hand-tuned Metal FlashAttention.

## FlashAttention: faster than PyTorch, competitive with MLX

Routed the way it now ships, FlashAttention runs faster than PyTorch's SDPA in every case I measured, and the numbers are a few paragraphs down. That took no change to the kernel, only to how it gets dispatched. When I first benchmarked the path that actually shipped, a plain `@triton.jit` call end to end, it was running at 0.3× SDPA, so before touching the math I went looking for why.

The kernels run through `torch.mps.compile_shader`, which takes MSL source and runs the kernel on the MPS tensor's own GPU buffer with no host round trip. The bug was in that zero-copy path. The simdgroup FlashAttention kernel launches on a two-dimensional threadgroup grid, and my dispatch only sent one-dimensional grids down the fast path, so every attention call quietly fell back to a host-round-trip path that is two to four times slower for reasons that have nothing to do with the arithmetic. It had been doing that the whole time. The earlier parity figure was measured on that same slow path, so the reading is not that SDPA got faster; the shipped path carried a routing bug I had not caught.

The fix was a one-line change to the dispatch predicate, and a clean A/B confirmed the diagnosis. The same kernel, launched on its native two-dimensional grid and then on a flattened one-dimensional grid, measured within a rounding error of itself, 6.32 against 6.31 TFLOP/s. The whole gap was the routing, not the grid and not the code. Once it was routed through the zero-copy path, with no change to the kernel, fp16 full attention ran at 1.65 to 1.99× SDPA and fp32 at 1.27 to 1.53×. That held at head dimensions 64 and 128, causal and not.

![The shipped path had been falling back to a slow host round-trip. Routing it through the zero-copy dispatch, with no change to the kernel, put it past SDPA.](/blog/triton-msl-part-2/dispatch.svg)

*The shipped path had been falling back to a slow host round-trip. Routing it through the zero-copy dispatch, with no change to the kernel, put it past SDPA.*

Two more changes made causal attention fast, and both are exact: neither changes the result by a single bit against the fp32 reference. The first is to stop computing masked work. A causal query block only attends to key blocks up to its own diagonal, but the kernel was walking every key block and masking off roughly half; stopping the loop at the diagonal is safe, because the per-element mask still handles the one partially-covered block. That took causal to 4× SDPA.

The second is to parallelize the softmax, which had been running on 32 of the 256 threads while the other 224 sat idle. Spreading it across all 256, eight per row, is again exact, because masked columns become negative infinity in the max pass and so contribute zero in the sum pass. Causal picked up about 80 percent from that, because once the masked work is skipped the softmax is a much larger share of what remains. The net result is causal attention up to 4× PyTorch's SDPA. Measured against MLX's own causal kernel, it lands exactly where full attention does: about even at moderate sequence lengths, and roughly ten percent behind at the largest, for the reason I get to below.

## Latent attention, the 2026 frontier

Several of the newer models this year, DeepSeek-V3 and Kimi among them, do not use plain multi-head attention. They use multi-head latent attention, whose score comes from two matmuls over different head dimensions, summed: a 128-wide "nope" part and a 64-wide "rope" part. The query-key contraction is therefore 192 wide, while the output stays 128 wide.

A symmetric 192-wide head does not fit the M-series register file and will not even compile. But latent attention is asymmetric, and the register footprint is set by the output width, not the contraction. The same simdgroup kernel handles it. A real nope/rope kernel written in `@triton.jit` now auto-routes: the compiler recognizes the two chained query-key dot products. It concatenates the split halves into one contiguous 192-wide operand and runs the fast kernel. That matches SDPA, and it runs 1.19 to 1.59× faster. As far as I know, this is the only way to run latent attention on Metal from a Triton kernel today.

## Where full attention stands, and why

Full and causal attention both run at about 0.88× MLX at the largest sizes, the one place MLX is still ahead, and I can account for that 12 percent exactly. The kernel is latency-bound on its device loads, and the current design already hides that latency about as well as MSL can express. Getting to that answer took profiling and four experiments.

The first job was to find where the time goes. My own design notes claimed the kernel was register-bound; pipeline reflection disagreed, reporting the maximum threads per threadgroup as 1024 with no cap applied. Occupancy was the next suspect, but shrinking the kernel's threadgroup memory from 26 KB to 7 KB, which should let several more groups run per core, moved throughput by 4 percent. The sharpest probe was an opt-in fp16-accumulate mode that halves the cost of the matmuls, and it bought about 4 percent of wall-clock. Since the half-precision matrix path on M4 runs at 1.38× the float one, a change that large moving the total that little means the matmuls are only about 14 percent of the runtime. The attention math is not the bottleneck.

![Halving the matmul cost bought only 4 percent, which places the matmuls at roughly 14 percent of the runtime. The bottleneck is everything else.](/blog/triton-msl-part-2/breakdown.svg)

*Halving the matmul cost bought only 4 percent, which places the matmuls at roughly 14 percent of the runtime. The bottleneck is everything else.*

So the other 86 percent was the target, and two experiments settled it. I halved the number of probability-fragment loads in the inner loop, which were being reloaded once per output tile for no reason. The output was bit-for-bit identical, and the kernel was 30 percent slower. The reason is that those redundant-looking loads were feeding a prefetch that overlaps device-load latency with compute; removing them exposed the latency. That one result reframed the problem. The kernel is latency-bound on its device loads, and the prefetch is load-bearing. I then added an explicit prefetch to the one load that lacked it, the transposed key load, and lost 3 percent, because the Metal compiler already schedules that load well and my version only got in its way. The last idea was structural: tiling 64 query rows per threadgroup instead of 32 would amortize the softmax and the rescale over twice the rows, but at head dimension 128, staging 64 rows of the query blows the 32 KB threadgroup budget and forces the query to be read from device memory. Since the kernel is latency-bound on exactly those reads, the trade loses; I built it, confirmed it correct, and measured it 20 to 25 percent slower.

![Every source-level lever is a dead end. Three make it slower and the fourth trades accuracy for 4 percent, because the wall is latency-hiding, which the current design already does about as well as MSL can express.](/blog/triton-msl-part-2/experiments.svg)

*Every source-level lever is a dead end. Three make it slower and the fourth trades accuracy for 4 percent, because the wall is latency-hiding, which the current design already does about as well as MSL can express.*

None of the four helped, and they point in the same direction: the kernel spends its time waiting on device loads, and the current layout already hides most of that wait about as well as I know how to in MSL. The matmul had pushed me toward the same conclusion earlier. It got faster when I dropped the shared-memory staging my first version used and loaded the fragments straight from device, letting the cache supply the reuse. The moves I reached for out of NVIDIA habit went the other way here: bigger tiles lost on occupancy, and double-buffering was a wash.

Attention looks like the same story. The thing that would actually help is an asynchronous copy engine that moves tiles while the cores compute, and that is not available here, for reasons in a later section. So on an M4 this design seems to top out around 0.88×. I have not found a source-level change that improves it, and the four I tried are the evidence for that rather than a claim that none exists.

That also settles a question I had been holding open. For a while I assumed attention might have to lean on a hand-tuned library kernel rather than being generated, since the gap looked real and most of attention is not doing anything Triton-specific. The dispatch fix changed that: the generated kernel came out ahead of PyTorch in every case I measured and competitive with MLX, and the one place it still trails MLX is something I could at least measure and explain.

## Quantized inference

The other thing that shipped is not attention. Weight-only int8 and int4 matmuls now auto-route to dedicated dequantizing kernels, in both the natural `[K, N]` layout and the GPTQ-style `[N, K]`.

The real win is decode. An int8 weight-only GEMV runs at the memory roofline, about 3.7× faster than the fp32 decode, because it moves a quarter of the bytes. Prefill is a different story, where the win is a smaller memory footprint rather than raw speed, since fp32 MPS BLAS still runs it faster. Alongside those, int4 gained per-group decode with zero points, and the skinny, deep matmul shapes that were starved for occupancy gained a deterministic two-pass split-K.

![Decode is memory-bound, so moving a quarter of the bytes is most of the win. Prefill is where the fp32 BLAS still leads, and the benchmark says so.](/blog/triton-msl-part-2/int8-decode.svg)

*Decode is memory-bound, so moving a quarter of the bytes is most of the win. Prefill is where the fp32 BLAS still leads, and the benchmark says so.*

All of it holds the same contract as the rest of the backend: if a shape or dtype falls outside what the fast kernel handles, it refuses loudly rather than return wrong numbers. That contract earned its keep here by catching a genuine silent-wrong, a loop-carried two-dimensional reduce that had been collapsing every output row onto the first.

## macOS 26: a non-event, by design

The durability argument in the first post was a specific engineering choice: emit public MSL text and let Apple's compiler lower it, rather than emitting the private AIR underneath. AIR is not a stable public surface, and its ABI shifts between OS releases, so owning the lowering to it would mean owning every one of those breaks. Targeting MSL, the surface Apple supports on purpose, is what keeps an OS bump away from codegen. macOS 26, Tahoe, was the first release to exercise that choice, and the pointed question I got after the post named a concrete risk: Apple restricts a private asynchronous simdgroup-copy intrinsic on 26, and readers wanted to know whether the backend leaned on it.

It does not, and never did. Asynchronous copies are lowered to synchronous ones from the start, so that intrinsic never appears in the output on either path. A restriction on an op you do not emit is a non-event by construction, not a near miss. It is a real trade, though: the synchronous copy is part of why attention is latency-bound, and the asynchronous copy engine that would close some of that gap is the one the OS puts off limits, so it is unavailable regardless of what I emit.

What actually needed handling on 26 was operational, not architectural. Tahoe unbundled the Metal compiler into a separate two-gigabyte download that ships on demand, and a suite that had been passing 1971 cases showed 1251 failures between two runs on the same machine, because Xcode had auto-updated and dropped the toolchain underneath it. The backend now recognizes that exact signature and points you at the one command that restores it. Tahoe also moved the SDK version from 15 to 26, which broke the version-string parsing in PyTorch's MPS backend; this backend never parses that number, and instead probes each `-std=metalX.Y` by compiling a trivial kernel and keeping whatever loads. The release cost zero codegen changes, which is the whole reason to target the public surface.

## A third vendor, byte-identical

The first post verified byte-identical output on NVIDIA and Apple. It is now three, and a third vendor matching is the expected consequence of how the IEEE path is built, not a surprise. On that path Triton fixes the reduction order and uses IEEE-conformant ops, so the result is a deterministic function of the program rather than of the hardware. Two backends agreeing followed from that; a third agreeing is the same property holding again. Running the same three kernels on AMD under ROCm, the vector-add and the ieee matmul produce output identical to the Metal and CUDA runs, down to the SHA-256, and the softmax matches to floating-point rounding.

![The same @triton.jit source, lowered to three different instruction sets by three Triton backends, produces output that hashes the same, and matches a NVIDIA run captured weeks earlier.](/blog/triton-msl-part-2/portability.svg)

*The same @triton.jit source, lowered to three different instruction sets by three Triton backends, produces output that hashes the same, and matches a NVIDIA run captured weeks earlier.*

One kernel source, three instruction sets, three Triton versions, and the hashes still line up. Portability is easy to claim and harder to show. This is a hash you can check.

## Where this leaves things

Attention is the part of this backend I would have hesitated to show six months ago. Now it is the first thing I would point at, and the only figure that still favors MLX is the gap at the largest sizes, which I accounted for above. The quieter work landed just as cleanly: quantized decode runs at the memory roofline, a third vendor's output hashes identically to the first two, and a major OS release came and went without a single codegen change.

A few threads are further along than this post but not ready to write up. Gluon, Triton's lower-level language, surfaced a real hole in its backend contract that kept it from running here at all; the fix is small and not Metal-specific, so it is drafted upstream, and a proper Gluon backend is its own project. The larger direction is where attention itself is going: the newest models are adopting linear and delta-rule variants, like the gated DeltaNet in Kimi's stack, which are not standard softmax attention. Getting those onto Metal, along with the backward pass that makes attention trainable, is where I went after this post. Both run now, though both are early enough that they earn their own writeup rather than a paragraph here.

If you want to try it, the install and the repo, github.com/bledden/triton-msl, are the same as last time. It is still alpha, and the standing request holds: the backend is built to refuse rather than miscompute, so if you ever catch it handing back a wrong number instead of an error, that is the bug I most want to hear about.

Same as last time: develop on the Mac you already own, and run the identical kernel on datacenter silicon.
