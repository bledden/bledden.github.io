---
title: "Same Source, Same Bytes: A Triton Metal Backend, Verified on NVIDIA"
description: "A Metal backend for OpenAI Triton. The same @triton.jit source, verified byte-identical on NVIDIA, with a correct-or-refuse contract. Develop correctness locally, tune performance on the target."
pubDate: 2026-07-15T12:00:00Z
heroImage: '../../assets/triton-msl-hero.png'
---

Triton is meant to be backend-agnostic. You write `@triton.jit`, the frontend lowers it to Triton IR (TTIR, then TTGIR), and a *backend* maps that program onto hardware. For years, though, "backend" meant NVIDIA. The only mature target was PTX, so writing Triton meant owning a CUDA GPU.

**triton-msl** is a Metal backend for Apple Silicon. The `@triton.jit` source and most of the compiler stack are shared across backends. What each backend owns is the lowering: the layout decisions, the hardware intrinsics, and the final code generation that map the program onto Metal or CUDA.

![The Python frontend, the Triton IR, and the optimization passes are shared. Each backend then applies its own layout choices, hardware intrinsics, and final codegen. Only those last stages differ between Metal and CUDA.](/blog/triton-msl/diagram-pipeline.svg)

*The Python frontend, the Triton IR, and the optimization passes are shared. Each backend then applies its own layout choices, hardware intrinsics, and final codegen. Only those last stages differ between Metal and CUDA.*

Triton's architecture is pluggable by design; it's how the AMD and Intel backends exist, and triton-msl is the Apple member of that family. With the source and most of the compiler shared, a kernel ought to be portable. Whether it actually produces the **same bits** on both backends is worth measuring rather than assuming. It does, and here is how I checked.

## Portability, measured

I took three kernels (vector-add, fused softmax, and a tiled matmul), ran the **identical Triton source** on an Apple M4 Max under Metal and a rented NVIDIA A40 under CUDA, and compared the raw output bytes. Here is the matmul, unchanged between the two runs:

```python
@triton.jit
def matmul(a_ptr, b_ptr, c_ptr, M, N, K,
           sam, sak, sbk, sbn, scm, scn,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
    b = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, tl.cdiv(K, BK)):            # the BK loop fixes the reduction order
        m = offs_k < K - k * BK
        acc += tl.dot(tl.load(a, mask=m[None, :], other=0.0),
                      tl.load(b, mask=m[:, None], other=0.0),
                      input_precision="ieee")
        a += BK * sak; b += BK * sbk
    c = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    tl.store(c, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

| kernel | elements | bytewise-identical Mac vs A40 | max \|Δ\| | max ULP |
|---|---|---|---|---|
| vector_add | 98,432 | 98,432 · 100% | 0 | 0 |
| matmul fp32 / ieee | 65,536 | 65,536 · 100% | 0 | 0 |
| fused_softmax | 99,968 | 27,440 · 27% | 7.45e-9 | 8 |

*Raw fp32 output bytes compared directly. vector_add and the `ieee` matmul are **byte-for-byte identical** across vendors, with matching SHA-256 (below). softmax differs on 72,528 of 99,968 elements, by at most 8 ULP or 7.45e-9, confined to `exp` and normalization rounding rather than the kernel's reduction structure.*

**Reproducibility receipt**

- **Mac** — M4 Max · macOS 15.6.1 · triton-msl 0.1.0a2 · Triton 3.7.0 · torch 2.12.1
- **A40** — NVIDIA A40 · CUDA 12.4 (driver 580.126.09) · Triton 3.7.0 · torch 2.4.1+cu124
- **Inputs** — np.random.seed(0), fp32 · vector_add n=98,432 · softmax 128×781 · matmul 256³ (BM=BN=64, BK=32, input_precision="ieee")
- **Compare** — raw fp32 bytes, SHA-256
- **vector_add** — 98,432 / 98,432 identical sha256 0fee55ab3218 == 0fee55ab3218
- **matmul(ieee)** — 65,536 / 65,536 identical sha256 b650d0b7667f == b650d0b7667f
- **softmax** — 72,528 / 99,968 differ ≤ 8 ULP / 7.45e-9 (exp + normalize rounding)
- **Harness** — [benchmarks/cross_backend_verify.py](https://github.com/bledden/triton-msl/blob/main/benchmarks/cross_backend_verify.py)

*Verified on the A40 under Triton 3.7.0 (matching the Mac) *and*, separately, the image's Triton 3.0.0. Both were byte-identical, with the same SHA-256. This is a **controlled same-compiler-version comparison**, and the Triton release doesn't move the bytes.*

**What this shows**
- Identical Triton source produces identical fp32 *bits* across Metal and CUDA for vector-add and this matmul.
- The source-level tiling and reduction structure is portable.

**What it doesn't**
- Performance portability, which is hardware-specific (see below).
- Bit-identical transcendentals: softmax differs by up to 8 ULP.
- A proof that every accepted kernel is correct. This is three kernels, not a theorem.

Why is byte-identical even *possible* across two GPU architectures? Because Triton's execution model is explicit. You specify the tiling (`BLOCK_M`, `BLOCK_K`, and so on), and the kernel runs a fixed sequence of IEEE-754 operations in an order the *kernel* pins rather than the hardware; the `BK` loop above is what fixes the reduction order. For these kernels, with these compiler settings, fixing the tiling, the reduction order, and the IEEE fp32 dot path was enough for both backends to run an equivalent multiply-accumulate sequence, with the same rounding and therefore the same bits. That holds for the paths I measured. It is not a general cross-GPU IEEE guarantee; contraction, denormals, and instruction selection can still diverge elsewhere. Softmax is the exception, and the receipt says exactly how: 72,528 of 99,968 elements differ, all within 8 ULP, because `exp` and the normalizing divide round differently between the two math libraries. That is a transcendental-rounding difference, not a structural one.

There is one configuration-level divergence worth dwelling on. NVIDIA's `tl.dot` defaults to **tf32** for fp32 inputs, which cuts the effective input mantissa precision before the multiply (accumulation stays in fp32). That is a deliberate trade, not a defect: on this A40, tf32 makes the same matmul about 1.8× faster than ieee (32 versus 18 TFLOP/s at 2048³), which is why NVIDIA defaults to it. The wrinkle for a portability check is that the two backends default to different precision, so the *same* matmul lands well off the fp64 reference on the A40 until you pass `input_precision="ieee"`, at which point it becomes byte-identical with the Mac. Metal has no tf32, so triton-msl does true fp32 unconditionally and **refuses** an explicit `tf32` request rather than quietly approximating it:

![On the default path, the same matmul is about 1,300× less accurate on NVIDIA than on Metal. With the default `tl.dot` fp32 configuration in this test, the Mac gives the *more* precise result until NVIDIA opts into `ieee`. That is a chosen execution mode, not a claim about the silicon.](/blog/triton-msl/chart-precision.svg)

*On the default path, the same matmul is about 1,300× less accurate on NVIDIA than on Metal. With the default `tl.dot` fp32 configuration in this test, the Mac gives the *more* precise result until NVIDIA opts into `ieee`. That is a chosen execution mode, not a claim about the silicon.*

Reproduce the whole thing: [benchmarks/cross_backend_verify.py](https://github.com/bledden/triton-msl/blob/main/benchmarks/cross_backend_verify.py) · [PORTABILITY.md](https://github.com/bledden/triton-msl/blob/main/PORTABILITY.md)

This is the part that matters in practice. **Kernel logic (tiling, masking, the reduction structure, the numerics) is portable, so you can write and correctness-debug it on the machine already on your desk.** Performance is not portable, which is what shapes the workflow: **develop correctness locally, then tune performance on the target**. You iterate for free on the Mac you own and rent a GPU only for the performance pass. The verification run itself took about eight minutes of A40 time and a few cents; provisioning and environment setup are on top of that.

![Two loops, cleanly split. The correctness loop runs on the Mac: free, offline, and quick, since an edited kernel recompiles in a fraction of a second and an unchanged one runs straight from cache. The performance loop runs on a rented GPU (minutes, a few cents), and only once the logic is already right.](/blog/triton-msl/diagram-workflow.svg)

*Two loops, cleanly split. The correctness loop runs on the Mac: free, offline, and quick, since an edited kernel recompiles in a fraction of a second and an unchanged one runs straight from cache. The performance loop runs on a rented GPU (minutes, a few cents), and only once the logic is already right.*

## Correct-or-refuse: the contract that makes the rest trustworthy

None of the portability claim means anything if the backend silently miscompiles. A byte-identical result on three kernels tells you nothing if the fourth quietly returns garbage. So triton-msl is built around a plain, unglamorous contract: **an unsupported or unvalidated lowering path raises `MetalNonRecoverableError` instead of emitting a plausible-but-wrong result.** Differential tests, fuzzing, and CI exercise that contract continuously. This is alpha software, not a proof of correctness, but the boundary it draws is what makes the rest usable.

Why does the boundary matter so much? Because a subset you can trust is something you can build on: if it runs, it's within the validated surface, and if it can't, it fails loudly. A larger surface that is occasionally, silently wrong is a liability, because you can't tell which results to believe.

![Every op takes one of two exits: a verified-correct lowering, or a loud `MetalNonRecoverableError`. The differential fuzzer and CI hold the line on that invariant.](/blog/triton-msl/diagram-refuse.svg)

*Every op takes one of two exits: a verified-correct lowering, or a loud `MetalNonRecoverableError`. The differential fuzzer and CI hold the line on that invariant.*

Holding that line is where the real work goes. Here is one example, because it's the failure mode that actually bites.

> **The unsigned-reduction bug.** Triton has no signed/unsigned distinction in its integer *types*; `uint32` and `int32` both lower to a signless `i32`. Signedness lives in the *operations* instead: unsigned max is `arith.maxui`, signed max is `arith.maxsi`. An early reduce lowering keyed off the tensor dtype, saw `i32`, and emitted a *signed* max. For a `tl.max` over `uint32`, everything below 2³¹ is fine. Everything above 2³¹ is silently wrong, because as signed integers those values read as negative, so "max" returns the *smallest* unsigned value. Plausible output, wrong answer, no error.

That is invisible to any test that only uses small values. What caught it was a **differential fuzzer**: generate random reduce kernels over random dtypes and value ranges, run each through the Metal backend *and* a reference, and assert the invariant `correct OR loud-refuse`. Feed it `uint32` values past 2³¹ and the Metal output diverges, and the run fails hard. The fix keys off the *combine op* (`maxui` versus `maxsi`) instead of the dtype name, and it generalized to a whole class: `uint64` max/min, and unsigned `argmin` and `argmax`, the same trap at 64 bits.

That is one of roughly **75 silent-wrongs** closed across the dot, reduce, and store surfaces over the alpha: masked stores that dropped a tighter-than-tile mask, stride inference that misread a transposed operand, a reduce+scan fusion that computed the scan and quietly dropped the reduce. Each is now either correct or a loud refusal, verified by differential and fuzz harnesses in CI. That is what separates a backend you can build on from one you can't.

Here is the surface in full: what computes, and where it draws the line.

| surface | computes | refuses (or warns) |
|---|---|---|
| matmul dtypes | fp32, fp16, and bf16 on the simdgroup matrix units | tf32 (only `ieee` is offered); fp64 → fp32 downcast + warning |
| matmul patterns | 2-D incl. transposed / sliced / column-major operands | chain-dot, batched / 3-D dot, un-inferable strides |
| FlashAttention | head_dim 32 / 64 / 128, causal + non-causal, fp32 / fp16 | bf16; head_dim ∉ {32, 64, 128} |
| reductions | sum/prod/max/min/and/or/xor, argmin/argmax, Welford, nan-max/min; signed *and* unsigned correct | unrecognized combine (never silently sums); i64 argmin/argmax |
| scans | cumsum / cumprod, forward + reverse, 1-D / 2-D, up to 64-bit | tiles > 1024 elements |
| atomics | add/max/min/and/or/xor/exch on i32 / u32 / fp32 / fp16 / bf16 | i64 / u64 (Metal has no 64-bit device atomic) |
| elementwise | broad: arith, math, select, casts, clamp, … | — |

*The supported surface, and where it draws the line. "Refuses" means a loud `MetalNonRecoverableError`, not a wrong answer. There is one deliberate exception: fp64, which the hardware can't do, downcasts to fp32 with a warning rather than refusing outright.*

## What works, and where the edges are

| Metric | Meaning |
|---|---|
| **2/3** | example kernels byte-identical on NVIDIA |
| **5,560/0/3,782** | upstream test_core: passed / failed / skipped |
| **~75** | silent-wrong bugs closed to correct-or-refuse |
| **1,968/0** | project suite pass / fail |

It's alpha, but the surface is real. On an M4 Max, **matmul** runs on the Apple matrix units and tracks PyTorch's own native-Metal matmul closely, coming within a few percent at 2048³. That tracks the native library rather than the hardware ceiling; hand-tuned Metal and MLX push some kernels higher.

![Median of 50 timed iterations after 15 warmups, with `torch.mps.synchronize()` around each; fp32, row-major, square M=N=K; PyTorch 2.12.1 `torch.mm` on an MPS tensor, no `torch.compile`; triton-msl at BM=BN=64, BK=32. Whiskers show min to max. Both are thermally sensitive on the M4 Max, so these are warm-machine medians. Read them as "tracks native within a few percent," not as a fixed peak.](/blog/triton-msl/chart-throughput.svg)

*Median of 50 timed iterations after 15 warmups, with `torch.mps.synchronize()` around each; fp32, row-major, square M=N=K; PyTorch 2.12.1 `torch.mm` on an MPS tensor, no `torch.compile`; triton-msl at BM=BN=64, BK=32. Whiskers show min to max. Both are thermally sensitive on the M4 Max, so these are warm-machine medians. Read them as "tracks native within a few percent," not as a fixed peak.*

The matrix units take fp16 and bf16 as well as fp32, and all three land within a hair of each other at 2048³:

![Matmul throughput by input dtype at 2048³, all accumulating in fp32. fp16 lowers to `simdgroup_half8x8` and bf16 to `simdgroup_bfloat8x8`. The common assumption is that Apple has no bf16 matrix unit; on the M4 that is simply not the case.](/blog/triton-msl/chart-dtypes.svg)

*Matmul throughput by input dtype at 2048³, all accumulating in fp32. fp16 lowers to `simdgroup_half8x8` and bf16 to `simdgroup_bfloat8x8`. The common assumption is that Apple has no bf16 matrix unit; on the M4 that is simply not the case.*

### Where Metal pays off: memory-bound work

The matrix units are only half the story. For memory-bound kernels like elementwise ops, softmax, and reductions, what matters is bandwidth. Metal's unified memory lets triton-msl run them *zero-copy* on the same buffers PyTorch already holds, through `torch.mps.compile_shader`, with no host round-trip. That is an 8 to 18× jump over marshalling data across the boundary, and it puts these kernels at 40 to 64% of the M4 Max's roughly 546 GB/s peak:

![Achieved bandwidth, zero-copy versus a host round-trip, for four memory-bound kernels (16M fp32 elements). Zero-copy is the default path; the host path is what you'd get marshalling buffers in and out. The dashed line is the M4 Max's roughly 546 GB/s peak.](/blog/triton-msl/chart-bandwidth.svg)

*Achieved bandwidth, zero-copy versus a host round-trip, for four memory-bound kernels (16M fp32 elements). Zero-copy is the default path; the host path is what you'd get marshalling buffers in and out. The dashed line is the M4 Max's roughly 546 GB/s peak.*

There is more than matmul. A **FlashAttention-2-style** kernel (causal and non-causal, head_dim 32, 64, or 128) runs through a simdgroup-MMA template at roughly 6 to 9.5 TFLOP/s at head_dim 128, from fp32 to fp16. That is a large speedup over the scalar path, though not competitive with hand-tuned Metal attention or MLX in absolute terms. Under the supported compilation config, **`torch.compile`** can route both inference and training graphs through triton-msl; that path is validated across 32 models (transformer blocks, a small GPT, a ViT, ResNets, LSTMs) and correct to eager within floating-point tolerance. At small-model sizes its compiled latency is roughly on par with eager MPS, so the value there is coverage, not speed. And it passes the upstream Triton `test_core` suite at **5,560 passing test cases, 0 failures, and 3,782 categorized skips** (reconciled test by test, almost all hardware-bounded: no fp8/mxfp matrix unit, no fp64, no 64-bit atomics, no tf32, or over Metal's threadgroup limits, and effectively none are unwritten lowerings). A skip here is the contract working: unsupported work is refused, not bluffed.

![The head_dim-128 FlashAttention-2 kernel on the matrix units. Q is staged once into registers, then a single loop over K/V tiles runs QKᵀ, then online softmax (running max and sum, with a rescale), then P·V, keeping the output accumulator register-resident the whole way.](/blog/triton-msl/diagram-fa.svg)

*The head_dim-128 FlashAttention-2 kernel on the matrix units. Q is staged once into registers, then a single loop over K/V tiles runs QKᵀ, then online softmax (running max and sum, with a rescale), then P·V, keeping the output accumulator register-resident the whole way.*

And those 3,782 skips aren't hidden failures. The harness turns anything that would silently miscompute into a hard refusal, which counts as a *failure*, of which there are zero. A skip is therefore always out-of-scope work, never a wrong answer swept under the rug. They break down like this:

![The 3,782 upstream `test_core` skips by reason, reconciled test-by-test. Almost all are hardware. Apple's simdgroup matrix units are fp16/bf16/fp32 only, so block-scaled mxfp/fp8 microscaling matmul (the single largest block) has no unit; likewise no fp64, no 64-bit atomics, no tf32. A further slice needs more than Metal's 1024 threads or 32&#8202;KB of threadgroup memory per group. The rest is CUDA-only test scaffolding. Reconciled test by test, effectively none are unwritten backend lowerings.](/blog/triton-msl/chart-skips.svg)

*The 3,782 upstream `test_core` skips by reason, reconciled test-by-test. Almost all are hardware. Apple's simdgroup matrix units are fp16/bf16/fp32 only, so block-scaled mxfp/fp8 microscaling matmul (the single largest block) has no unit; likewise no fp64, no 64-bit atomics, no tf32. A further slice needs more than Metal's 1024 threads or 32&#8202;KB of threadgroup memory per group. The rest is CUDA-only test scaffolding. Reconciled test by test, effectively none are unwritten backend lowerings.*

Three honest limits. **Performance is not portable.** The same matmul that does 11.3 TFLOP/s on the M4 Max does 18.2 in fp32, and 32.2 in tf32, on the A40 at 2048³. Those are the same tutorial kernel on both machines, not either one's library-tuned peak; a real cuBLAS or cutlass path pulls the A40 well past 32. Either way the NVIDIA card is faster, as you would expect, and that gap is exactly why performance work belongs on the target rather than the laptop. Block sizes, occupancy, and fast-path routing are hardware-specific too, so an M4's optimal tiling tells you nothing about an H100's. Develop *correctness* locally, tune *performance* on the target. **The tested surface is one machine.** Everything here was measured on an M4 Max; the backend runs on M1 through M4 via the simdgroup-matrix path, with bf16 falling back to fp32 on parts that have no bfloat matrix unit, but per-chip CI is not in place yet, so treat older parts as should-work rather than verified. And **the supported set is a subset**, but the boundary is mostly hardware, not backlog. Larger attention exceeds Metal's threadgroup-memory budget, block-scaled mxfp has no matrix unit, and reconciling the skipped conformance tests one by one turns up almost no unwritten lowerings. What the hardware can't do refuses loudly.

## Try it

```bash
pip install triton-msl        # import triton_msl
pip install "triton>=3.6.0"   # Triton itself, installed separately
```

One caveat up front: Triton publishes no macOS wheel, so on a Mac you build it from source (`pip install git+https://github.com/triton-lang/triton.git`). That one-time build, about twelve minutes, is the real setup cost; after it, triton-msl is a normal import with no build of its own. If you happen to be on the exact tuple I build against, Python 3.14 and macOS 15+ on Apple Silicon, there is an unofficial prebuilt Triton wheel on the [repo's releases](https://github.com/bledden/triton-msl/releases) that skips the build entirely.

The repo, the full portability receipt with its reproduce harness, and the correctness campaign in detail: [github.com/bledden/triton-msl](https://github.com/bledden/triton-msl).

If you write Triton and own a Mac, the most useful thing you could do is try to make it silently return a wrong number. By the contract, it shouldn't be able to. If it does, that's the bug report I most want.
