---
title: 'The Distribution Cliff: Why Hybrid Distillation Catastrophically Fails'
description: 'We tested 8 distillation methods across 160 runs. Any off-policy component causes collapse. Pure on-policy with teacher seeding achieves 71% GSM8K accuracy.'
pubDate: 2026-01-02
heroImage: '../../assets/context-distillation-hero.png'
---

The GKD paper recommends hybrid distillation: bootstrap with off-policy, then refine with on-policy. We tested this across 160 training runs with two model families.

**It catastrophically fails.** Every method with any off-policy component collapses to 0% accuracy.

---

## The Research Question

From the [original project spec](https://github.com/thinking-machines-lab/tinker-project-ideas/blob/main/on-policy-context-distillation.md):

> How do different context distillation methodologies compare in their effectiveness for training student models to match teacher model performance?

The spec proposed comparing three approaches for **context distillation** (same model with/without few-shot context):
1. Off-policy distillation alone
2. On-policy distillation alone
3. Hybrid: off-policy → on-policy

This builds on foundational work by [Anthropic (2021)](https://arxiv.org/abs/2112.00861) and [Snell et al. (2022)](https://arxiv.org/abs/2209.15189), with on-policy methods from [Agarwal et al. (2023)](https://arxiv.org/abs/2306.13649).

**What we discovered**: The answer depends critically on whether there's a capability gap between teacher and student—and this changes everything.

---

## The Distribution Cliff

When transitioning from off-policy to on-policy training:

```
Step 0-49:   Off-policy (supervised on teacher outputs)
Step 50:     PHASE TRANSITION
             → Student generates with modified weights
             → Produces "teacher-like garbage"
             → Teacher assigns low logprobs
             → GKD pushes toward degenerate solutions
Step 50-100: COLLAPSE (0% accuracy, unrecoverable)
```

![Distribution Cliff](/images/context-distillation/distribution_cliff.png)

The collapse happens in one step. Scores drop from ~0.5 to 0 and never recover.

---

## Why It Happens

Off-policy and on-policy training optimize fundamentally different objectives:

| Training Mode | Objective | What Student Learns |
|---------------|-----------|---------------------|
| Off-Policy | max P(teacher_token \| prefix) | Token prediction (mimicry) |
| On-Policy GKD | min D_KL(student ‖ teacher) | Distribution matching |

When the student is much smaller than the teacher:

1. **Off-policy phase**: Student learns to predict teacher's tokens. Weights shift toward teacher's vocabulary patterns. But without the reasoning capacity to understand *why*.

2. **Phase transition**: Student generates with modified weights. Produces tokens that "look like" teacher output (similar vocabulary, formatting) but are semantically incoherent.

3. **On-policy phase**: Teacher evaluates garbage. Assigns very low log-probabilities. GKD computes large negative advantages. Student collapses to empty/repetitive outputs.

**The key insight**: Off-policy teaches *token mimicry* without *reasoning*. When forced to generate independently, the student produces syntactically similar but semantically broken output.

---

## The 160-Run Experiment

We tested 8 distillation methods across two model families:

| Family | Student | Teacher | Size Ratio |
|--------|---------|---------|:----------:|
| **Qwen** | Qwen3-4B | Qwen3-30B-A3B | 7.5x |
| **Llama** | Llama-3.1-8B | Llama-3.3-70B | 8.75x |

**Methods tested**:
- `on_policy_gkd` - Pure on-policy (baseline)
- `hybrid` - Off-policy → on-policy transition
- `extended_on_policy` - More training steps
- `teacher_seeded` - Decaying teacher prefix
- `mixture` - Blend objectives each step
- `replay_buffer` - Experience replay
- `kl_anchored` - KL penalty to initial weights
- `reverse_curriculum` - On-policy first, off-policy last

---

## Results: The Pattern is Unambiguous

### Qwen Family (4B ← 30B)

| Method | GSM8K Accuracy | Status |
|--------|:--------------:|--------|
| **teacher_seeded** | **58.6%** | Best |
| on_policy_gkd | 53.2% | Stable |
| extended_on_policy | 51.0% | Good |
| replay_buffer | 2.8% | Partial |
| hybrid | 0.0% | Collapsed |
| mixture | 0.0% | Collapsed |
| kl_anchored | 0.0% | Collapsed |
| reverse_curriculum | 0.8% | Collapsed |

### Llama Family (8B ← 70B)

| Method | GSM8K Accuracy | Status |
|--------|:--------------:|--------|
| **teacher_seeded** | **71.0%** | Best |
| on_policy_gkd | 67.4% | Stable |
| extended_on_policy | 64.4% | Good |
| replay_buffer | 7.4% | Partial |
| hybrid | 0.0% | Collapsed |
| mixture | 0.0% | Collapsed |
| kl_anchored | 0.0% | Collapsed |
| reverse_curriculum | 0.0% | Collapsed |

![Method Comparison](/images/context-distillation/method_comparison.png)

**Every method with any off-policy component collapses.** Pure on-policy methods work.

---

## The W&B Training Curves

The [full W&B report](https://wandb.ai/facilitair/context-distillation) shows the training dynamics across all 160 runs:

**Key observations from the curves:**

1. **`train/phase`** (hybrid runs): Shows the binary 0→1 transition at step 50. This is when collapse occurs.

2. **`train/score`**: Pure on-policy methods (green/yellow lines) maintain scores of 0.4-0.6 throughout training. Hybrid/mixture methods (other colors) crash to near-zero after the phase transition.

3. **`train/on_policy_loss`** (mixture): Goes deeply negative (-500 to -1000) as the GKD objective tries to correct garbage outputs with extreme gradients.

4. **`train/kl_divergence`**: Spikes to 4-6 during collapse as student distribution diverges catastrophically from teacher.

5. **`eval/avg_score`**: The clearest signal—on_policy_gkd runs cluster at 0.5-0.6, while hybrid runs flatline at 0-0.1.

6. **`train/seed_tokens`** (teacher_seeded): Shows the smooth decay from 20→0 tokens over 50 steps. No phase transition, no collapse.

The curves make the failure mode visually obvious: stable training → phase transition → immediate collapse → no recovery.

---

## What We Tried (And What Failed)

### KL Regularization

We added a KL penalty to prevent drift from initial weights:

```python
total_loss = off_policy_loss + beta * D_KL(current || initial)
```

**Result**: Still collapsed. The KL penalty slows drift but doesn't prevent the fundamental mimicry-without-reasoning problem.

### Mixing Objectives

We blended off-policy and on-policy signals each step:

```python
loss = 0.3 * off_policy_loss + 0.7 * gkd_loss
```

**Result**: Still collapsed. Even 30% off-policy signal is enough to corrupt generation over 100 steps.

### Experience Replay

We injected old teacher trajectories during on-policy training.

**Result**: Partial (2.8-7.4% accuracy). Better than pure hybrid, but still fundamentally broken.

### Reverse Curriculum

What if we do on-policy first, then off-policy refinement?

**Result**: Collapsed. The order doesn't matter—any off-policy exposure corrupts the student.

---

## Why Teacher Seeding Works

`teacher_seeded` provides off-policy *benefits* without off-policy's failure mode:

```
Step 0:   [Teacher: 20 tokens] [Student: rest]
Step 25:  [Teacher: 10 tokens] [Student: rest]
Step 50:  [Teacher: 0 tokens]  [Student: all]
```

- Teacher prefix keeps student generations coherent during warmup
- Student learns from meaningful teacher feedback (not garbage evaluation)
- Gradual handoff prevents the hard phase transition

![Teacher Seeding](/images/context-distillation/teacher_seeding.png)

**The key difference**: Student always generates something coherent. There's no moment where it produces garbage and gets catastrophically penalized.

---

## The Pivot: Context Distillation → Size Distillation

We started with the original spec's setup: **context distillation** using the same model with/without few-shot examples (following [Snell et al. 2022](https://arxiv.org/abs/2209.15189)).

### Phase 1: Same-Model Context Distillation (Failed)

We ran 10-seed experiments with Qwen3-4B as both teacher (with 10-shot context) and student (no context):

| Mode | Eval Score | Downstream Accuracy |
|------|:----------:|:-------------------:|
| Off-Policy | 0.97 | **0%** |
| On-Policy GKD | 0.44 | **2-6%** |
| Hybrid | 0.10 | **0%** |
| Hybrid Gradual | 0.13 | **0%** |

Off-policy achieved 0.97 eval score but 0% downstream accuracy. The student learned to **match the teacher's output format** but couldn't actually solve problems. On-policy GKD fared slightly better (2-6% accuracy) but still far below useful.

**Why**: When student and teacher are the same model, there's no capability gap. The few-shot context helps the teacher format answers correctly, but the student already has the same underlying reasoning ability. Distillation produces format matching, not capability transfer.

**Key insight**: Our eval metric (Jaccard + bigram similarity) was measuring format matching, not actual capability. This is why off-policy achieved 0.97 eval but 0% accuracy—the student matched the teacher's response style perfectly without learning to reason.

### Phase 2: Pivot to Size Distillation

To actually transfer capabilities, we switched to using larger teachers:

| Setup | Result |
|-------|--------|
| Qwen-4B ← Qwen-30B | 58.6% GSM8K accuracy |
| Llama-8B ← Llama-70B | 71.0% GSM8K accuracy |

This worked—but introduced the distribution cliff problem that became our main finding.

### Implications for the Original Spec

The original spec's premise—that context distillation can transfer few-shot learning ability—may be flawed:

1. **Context distillation is useful for format/style transfer**, not capability transfer
2. **The GKD paper's results may not generalize** to scenarios with large capability gaps
3. **Size distillation requires different techniques** than context distillation

For practitioners wanting to "distill" few-shot prompting ability: you're likely better off with retrieval-augmented generation or prompt caching than distillation.

---

## Implications for Practitioners

1. **Never use hybrid mode with capability gaps.** The GKD paper's recommendation doesn't generalize to size distillation.

2. **Use `teacher_seeded` for knowledge distillation.** Best results across both model families (+5.4 pp over pure GKD on Qwen, +3.6 pp on Llama).

3. **Pure `on_policy_gkd` is a safe baseline.** Works reliably, just slightly worse than teacher_seeded.

4. **Context distillation is for format/style, not capabilities.** Use size distillation for actual knowledge transfer.

---

## Cost Analysis

| Experiment | Runs | Duration | Cost |
|------------|:----:|:--------:|:----:|
| Qwen (4B ← 30B) | 80 | 62.7h | ~$47 |
| Llama (8B ← 70B) | 80 | 67.7h | ~$68 |
| **Total** | **160** | **130.4h** | **~$115** |

Cost per method validation (10 seeds): ~$7.20

The experiment was surprisingly cheap for the insight gained. The key finding—that off-policy corrupts generation—could save significant compute on future distillation projects.

---

## Answering the Original Spec

The [project spec](https://github.com/thinking-machines-lab/tinker-project-ideas/blob/main/on-policy-context-distillation.md) asked: *How do different context distillation methodologies compare?*

**For context distillation (same model ± context)**:
- Neither on-policy nor off-policy works well for capability transfer
- Off-policy achieves high format similarity (0.97 eval) but 0% downstream accuracy
- On-policy GKD fares slightly better (2-6%) but still far below useful
- Context distillation is useful for style/format, not reasoning ability

**For size distillation (small ← large model)**:
- On-policy significantly outperformed off-policy in our experiments
- Hybrid mode collapsed in all configurations we tested
- Teacher seeding achieved the best results (58-71% GSM8K accuracy)

**The deeper insight**: The GKD paper's hybrid recommendation assumes the student can generate coherent outputs after off-policy training. With large capability gaps, this assumption may not hold—the student learns to mimic tokens without understanding them, producing garbage when forced to generate independently.

**Caveats**: Our experiments used specific hyperparameters, model families, and a single benchmark (GSM8K). The collapse pattern was consistent across our tests, but different configurations (learning rates, longer warmup, different model architectures) might yield different results. Teacher seeding appears promising, but more investigation is warranted before drawing universal conclusions.

---

## Related Work

**Important context on the GKD paper**: [Agarwal et al. (2024)](https://arxiv.org/abs/2306.13649) tested GKD exclusively on **encoder-decoder models** (T5-small/base/large ← T5-XL, up to 38x size ratio). They did not evaluate decoder-only architectures like Llama or Qwen. Encoder-decoder models have fundamentally different generation dynamics—the encoder provides conditioning that may stabilize generation even after distribution drift. Our experiments extend GKD to decoder-only LLMs, where hybrid mode appears to fail catastrophically. This is not a contradiction of their results, but a finding that their hybrid recommendation may not generalize to modern decoder-only architectures.

The capability gap problem in distillation is well-documented. [Mirzadeh et al. (2020)](https://ojs.aaai.org/index.php/AAAI/article/view/5963) showed that "student network performance degrades when the gap between student and teacher is large," proposing Teacher Assistant models as intermediaries. [Speculative Knowledge Distillation](https://arxiv.org/abs/2410.11325) addresses this by dynamically adjusting on-policy vs off-policy balance based on the distribution gap.

Our contribution is documenting the specific failure mode at the phase transition in decoder-only LLMs—the instantaneous collapse when switching from off-policy to on-policy, with KL divergence spiking 361x in a single step. This appears to be a distinct phenomenon from gradual capability gap degradation, and may be specific to decoder-only architectures.

---

*160-run experiment across Qwen and Llama families using Tinker API. Full methodology at [github.com/bledden/context-distillation-tinkerideas](https://github.com/bledden/context-distillation-tinkerideas). W&B report at [wandb.ai/facilitair/context-distillation](https://wandb.ai/facilitair/context-distillation).*
