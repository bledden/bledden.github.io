---
title: 'Five Patterns From 250+ ML Evaluation Runs'
description: 'Cross-project analysis of ML experiments reveals consistent patterns about model scaling, mode collapse, and training efficiency.'
pubDate: 'Dec 31 2025'
heroImage: '../../assets/meta-analysis-hero.png'
---

After running 250+ ML evaluation runs across 6 independent projects, I started seeing patterns that repeated too consistently to ignore. This post documents five findings that emerged from that data, with exact numbers and source references.

## The Projects

The analysis spans experiments from:
- **Corch_by_Fac**: Foundation model training for classification
- **Facilitair_v2**: Multi-agent workflow orchestration
- **Tinker-experiments**: Constitutional training, GAN jokes, context distillation
- **Dendritic-hackathon**: Neuromorphic computing experiments
- **AMD_Hackathon**: Q&A agent fine-tuning
- **ArrwDB**: Vector database optimizations

---

## Finding 1: The "3B Cliff" — Capability Threshold

Models below ~3B parameters consistently fail at tasks that 8B models succeed at.

| Project | Task | 3B Performance | 8B Performance |
|---------|------|----------------|----------------|
| **Open Character** | Structured output (judging) | ~40% success | ~95% success |
| **Open Character** | Character distillation | 54% → 37% (regressed) | V2 pending |
| **GAN Jokes** | Originality score | 2.8/10 | V3 pending |

The structured output gap is particularly striking: **55 percentage points** difference just from model size. The 3B model's character training actually made it worse—a 17% regression—suggesting the model lacked capacity to learn the task.

![Model Size Capability Matrix](/images/meta-analysis/plot_5_3b_cliff_heatmap.png)

---

## Finding 2: Mode Collapse is Binary, Not Gradual

Mode collapse occurs **instantaneously** at a hyperparameter threshold, not as gradual degradation.

| Attempt | Learning Rate | Rank | Accuracy | Collapsed? |
|---------|--------------|------|----------|------------|
| 2 | 2e-4 | 128 | 2% ("10000000") | YES |
| 3 | 5e-5 | 64 | 0% ("10000000") | YES |
| 4 | 5e-6 | 32 | 73.5% | NO |

Just a **10x reduction** in learning rate (5e-5 → 5e-6) made the difference between complete collapse and successful convergence. Both 2e-4 and 5e-5 collapsed—the relationship is non-monotonic with a critical threshold.

![Mode Collapse Threshold](/images/meta-analysis/plot_4_mode_collapse_heatmap.png)

---

## Finding 3: Model Switching Beats Training (38x ROI)

When baseline accuracy is poor (<80%), switching models has dramatically higher ROI than training.

| Approach | Time | Accuracy | ROI (%/min) |
|----------|------|----------|-------------|
| Training Attempt 1 | 2h | 0-3% | -0.5 |
| Training Attempt 2 | 33m | 2% | -2.2 |
| Training Attempt 3 | 11m | 0% | -6.6 |
| Training Attempt 4 | 2.6m | 73.5% | +0.2 |
| **Model Switch** | **5m** | **92%** | **+3.8** |

**Total training time**: 2 hours 47 minutes for +0.5% improvement.
**Model switch time**: 5 minutes for +19% improvement.
**ROI ratio: 38x**

Qwen2.5-7B at 92% accuracy outperformed DeepSeek-R1-32B at 61-73%—4.5x smaller yet more accurate. I wasted 3 days optimizing the wrong model.

![Training vs Model Switch ROI](/images/meta-analysis/plot_2_roi_scatter.png)

---

## Finding 4: Hybrid Distillation Collapse

Hybrid distillation (off-policy → on-policy) causes complete collapse at the transition point.

**Exact log from Llama Hybrid (Seed 8)**:
```
=== Off-Policy Phase ===
Step 0:  loss=2321.64, eval_score=0.269
Step 40: loss=18.65,   eval_score=0.266

=== Transition to On-Policy ===
Step 50: loss=0.1053, score=0.000, kl=5.4453  ← COLLAPSE
Step 90: loss=0.1299, score=0.000, kl=4.3808
```

KL divergence spikes **361x** (0.015 → 5.4453) in one step at the exact moment of phase transition. The severity correlates with the capability gap:

| Gap | Family | Result | Severity |
|-----|--------|--------|----------|
| 70x (1B→70B) | Llama | → 0.00 | Complete |
| 7.5x (4B→30B) | Qwen | → 0.06-0.21 | Partial |

![Phase Transition Collapse](/images/meta-analysis/plot_3_phase_transition.png)

---

## Finding 5: Supervised Learning is 909% More Efficient Than RL

For memorization tasks, supervised learning dramatically outperforms reinforcement learning.

| N (values) | Supervised | RL-EoE | RL-Step | SL Advantage |
|:----------:|:----------:|:------:|:-------:|:------------:|
| 10 | **11 ep** | 101 ep | 6,040 ep | 9x / 549x |
| 100 | **11 ep** | 301 ep | 10,000+ ep | 27x / 909x+ |
| 500 | **11 ep** | 451 ep | 50,000+ ep | 41x / 4545x+ |

Statistical significance (Hedges' g):
- Supervised vs RL-EoE (N=10): **g = -8.2**, p < 0.001
- Supervised vs RL-Step (N=10): **g = -15.3**, p < 0.001

The scaling behavior is even more striking:

| Method | Exponent (β) | Time Complexity |
|--------|--------------|-----------------|
| Supervised | **0.00** | **O(1)** |
| RL-EoE | 0.35 | O(n^0.35) |
| RL-Step | 1.05 | O(n) |

Supervised learning converges in **constant time** regardless of problem size. RL-Step scales **linearly**.

![SL vs RL Scaling](/images/meta-analysis/plot_1_sl_vs_rl_scaling.png)

![SL vs RL at N=100](/images/meta-analysis/plot_6_sl_vs_rl_bars.png)

---

## Summary Dashboard

![Summary Dashboard](/images/meta-analysis/plot_8_summary_dashboard.png)

---

## Key Takeaways

1. **Model selection matters more than training** — Test broadly before committing
2. **Mode collapse has a threshold** — Find it early with small experiments
3. **The 3B cliff is real** — Some tasks require minimum model capacity
4. **Hybrid distillation is fragile** — The phase transition can destroy learning
5. **SL beats RL for memorization** — Use the right tool for the task

These patterns appeared independently across 6 projects over several months. They're not theoretical—they're what the data consistently showed.

---

*Data sources and exact file references available in the [full analysis](https://github.com/bledden/Corch_by_Fac/blob/main/CROSS_PROJECT_META_ANALYSIS.md).*
