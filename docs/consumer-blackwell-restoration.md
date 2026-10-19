# Consumer Blackwell article restoration

Restored October 18, 2026. Original publication dates retained.

{
  "restored_on": "2026-10-18",
  "original_urls": [
    "https://getpoppy.app/blog/writing-ptx-for-consumer-blackwell",
    "https://getpoppy.app/blog/four-months-upstreaming-consumer-blackwell"
  ],
  "author_html_sha256": "bb546f86bcba10b4933e45e329a011938b4a46a41c9ad089fc93514ce856dc5e",
  "preserved_images": [
    {
      "file": "ptx-original-01.png",
      "sha256": "5d0e1496e07f0b9d1241d7c5ff26d20ea13d6666637d7f6f4cd61364ecd00c46",
      "origin": "Embedded image in preserved local author HTML; published pixel identity not independently established."
    },
    {
      "file": "ptx-original-02.png",
      "sha256": "587ec8dad523446dbc020f348ca854513eaa857b6265261037d4a8117c237ded",
      "origin": "Embedded image in preserved local author HTML; published pixel identity not independently established."
    },
    {
      "file": "ptx-original-03.png",
      "sha256": "d8b9b89ecc708866322788f01ec17264fadfb86e5926519e906256ded433262c",
      "origin": "Embedded image in preserved local author HTML; published pixel identity not independently established."
    },
    {
      "file": "ptx-original-04.png",
      "sha256": "7f069a9651e359238cd202dcdce2b5bbf84a2d07bdd3a14bf0e4679361d2dc3d",
      "origin": "Embedded image in preserved local author HTML; published pixel identity not independently established."
    },
    {
      "file": "ptx-original-05.png",
      "sha256": "1182cd04e15a17a7b6d813f274e7c13a030a72a32325f2cdebac3d62a4af5cd4",
      "origin": "Embedded image in preserved local author HTML; published pixel identity not independently established."
    },
    {
      "file": "ptx-original-06.png",
      "sha256": "65450bff3e99075896b404d459cac6aea35980082a06edb60becbefd043aab71",
      "origin": "Embedded image in preserved local author HTML; published pixel identity not independently established."
    },
    {
      "file": "ptx-original-07.png",
      "sha256": "919aa9cce031ff18b2c2cfb2d3f49850c486bbb085dad512584114afe1df45d2",
      "origin": "Embedded image in preserved local author HTML; published pixel identity not independently established."
    },
    {
      "file": "ptx-original-08.png",
      "sha256": "f3940e63c5226debf50469efd9eb6132127607363ef78a951c438ad4130ea951",
      "origin": "Embedded image in preserved local author HTML; published pixel identity not independently established."
    },
    {
      "file": "ptx-original-09.png",
      "sha256": "d85e760191703ae0b1a8f520c73a9b42502e4c9a2d56c2ce99775e328b739b4f",
      "origin": "Embedded image in preserved local author HTML; published pixel identity not independently established."
    }
  ],
  "recreated_visuals": [
    "ptx-build-runtime.svg",
    "ptx-transpose-placement.svg",
    "ptx-stride-padding.svg",
    "upstreaming-decode-range.svg",
    "Upstreaming historical support table (HTML)"
  ],
  "benchmark_source": "https://github.com/Dao-AILab/flash-attention/pull/2634#issuecomment-4703768152",
  "limits": [
    "Not a byte-identical archive of published HTML.",
    "Original upstreaming images, raw per-sequence decode points, and companion report remain unavailable.",
    "Historical claims preserved; reconstructed visuals and added reference index are labeled."
  ]
}

References verified through GitHub issues API:

- https://github.com/sgl-project/sglang/pull/18747 — fix(sgl-kernel): support CUDA 13 runtime preloading for DGX Spark
- https://github.com/sgl-project/sglang/pull/18748 — fix(sgl-kernel): add SM12x to FA4 compute capability whitelist
- https://github.com/sgl-project/sglang/pull/18750 — fix(sgl-kernel): use >= 120 for SM12x CUDA kernel dispatch
- https://github.com/sgl-project/sglang/pull/18751 — fix: update Blackwell log/error messages to include SM12x
- https://github.com/sgl-project/sglang/pull/18754 — fix: add SM12x to JIT flash attention compute capability whitelist
- https://github.com/sgl-project/sglang/pull/18755 — fix: enable MXFP8 dense linear on SM12x Blackwell GPUs
- https://github.com/sgl-project/sglang/pull/18758 — fix: enable flashinfer backends and allreduce fusion for SM12x Blackwell GPUs
- https://github.com/flashinfer-ai/flashinfer/pull/2446 — feat: Add TRTLLM fmha_v2 library for SM90 attention with Skip-Softmax
- https://github.com/flashinfer-ai/flashinfer/pull/2559 — fix: allow fmha_v2_prefill_deepseek on SM121 (DGX Spark)
- https://github.com/flashinfer-ai/flashinfer/pull/2560 — fix: guard CUTLASS FMHA against SM12x and fix fmha_v2 SM121a check
- https://github.com/flashinfer-ai/flashinfer/pull/2561 — feat: SM120 standard attention kernels + CuTe DSL backend
- https://github.com/flashinfer-ai/flashinfer/pull/2574 — feat: add is_sm12x_supported() helper for SM12x family detection
- https://github.com/flashinfer-ai/flashinfer/pull/2598 — feat: add CuTe DSL flash attention backend for SM120 GPUs
- https://github.com/flashinfer-ai/flashinfer/pull/2654 — fix: Add fused MOE and GEMM AOT modules for SM121
- https://github.com/flashinfer-ai/flashinfer/issues/2655 — [Feature]: BF16 MLA decode support on SM120/SM121 (consumer Blackwell)
- https://github.com/flashinfer-ai/flashinfer/pull/2675 — Support BF16 MLA on SM120 with shared-mem fallback
- https://github.com/flashinfer-ai/flashinfer/pull/2689 — Fix 10 bugs in BF16 XQA MLA kernel for SM120/SM121
- https://github.com/flashinfer-ai/flashinfer/pull/3016 — feat: enable fmha_v2 HMMA attention for SM120 standard shapes
- https://github.com/flashinfer-ai/flashinfer/pull/3106 — fix(fmha_v2): enable flash_attention for Q_PAGED_KV regardless of s_kv
- https://github.com/NVIDIA/cutlass/pull/3030 — [CuTeDSL] Flash Attention v2 for SM120 (Blackwell GeForce)
- https://github.com/NVIDIA/cutlass/issues/3044 — CuTe DSL: FP8 MMA segfaults on SM120 — MmaAtomSM80Type missing kind::f8f6f4 lowering
- https://github.com/NVIDIA/cutlass/pull/3082 — fix: Use is_family_of() for SM12x arch guard in MmaSM120BlockScaledOp
- https://github.com/NVIDIA/cutlass/pull/3084 — fix: Use is_family_of() for SM90 arch guard in warpgroup MmaOp
- https://github.com/triton-lang/triton/pull/9572 — [Language] Add field inheritance, defaults, and aggregate_replace to @aggregate
- https://github.com/vllm-project/vllm/pull/35323 — Add Qwen3.5-MoE (qwen3_5_moe) support via Qwen3-Next backend
- https://github.com/vllm-project/vllm/pull/35568 — [Bugfix] Fix SM121 (DGX Spark) exclusion from Marlin/CUTLASS FP8 paths
- https://github.com/Dao-AILab/flash-attention/pull/2268 — Add SM120 (Blackwell GeForce / DGX Spark) flash attention
- https://github.com/Dao-AILab/flash-attention/pull/2325 — Fix SM80 forward+backward API drift in CuTe DSL kernels
- https://github.com/Dao-AILab/flash-attention/pull/2329 — SM120 forward pass (Blackwell GeForce / DGX Spark)
- https://github.com/Dao-AILab/flash-attention/pull/2330 — [Bwd,Sm120] Add SM120 backward pass support
- https://github.com/Dao-AILab/flash-attention/pull/2333 — Add SM120 varlen attention support
- https://github.com/Dao-AILab/flash-attention/pull/2336 — Add SM120 split-KV (FlashDecoding) with FP32 partial outputs
- https://github.com/Dao-AILab/flash-attention/pull/2348 — Add SM120 kernel-level paged KV cache support
- https://github.com/Dao-AILab/flash-attention/pull/2349 — Add SM120 TMA forward kernel with warp specialization
- https://github.com/Dao-AILab/flash-attention/pull/2389 — Add SM80/SM120 block-sparse forward attention support
- https://github.com/Dao-AILab/flash-attention/pull/2404 — fix(cute): SM120 forward/bwd and atomic add compatibility
- https://github.com/Dao-AILab/flash-attention/pull/2406 — [CuTe, SM120] Forward kernel with optimized TMA path and full features support
- https://github.com/Dao-AILab/flash-attention/pull/2416 — Fix SM120 forward pass crash: parent __init__ overwrites arch, enabling unsupported TMA path
- https://github.com/Dao-AILab/flash-attention/pull/2420 — [CuTe, SM120] Fix forward/backward bugs and add SM120 test guards
- https://github.com/Dao-AILab/flash-attention/issues/2436 — DropOut Cute DSL
- https://github.com/Dao-AILab/flash-attention/pull/2439 — Add dropout support to CuTe DSL attention kernels
- https://github.com/Dao-AILab/flash-attention/pull/2634 — FA4 consumer Blackwell (sm_120) integration
