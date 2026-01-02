---
title: 'Constitutional AI from Base Models: Can You Train Safety Without Instruction Tuning?'
description: 'We replicated Constitutional AI starting from a raw base model. The pipeline works, but DPO needs more than 42 training pairs to show improvement over SFT alone.'
pubDate: 'Jan 01 2026 06:00'
heroImage: '../../assets/cai-hero.png'
---

Constitutional AI (CAI) is Anthropic's approach to training helpful, harmless AI systems using a set of principles—a "constitution"—rather than pure human preference data. But most implementations start from instruction-tuned models, which may introduce biases from that initial training.

**The question**: Can we train CAI starting from a raw base model, with no instruction-tuning contamination?

After 10 days of experimentation and ~300 training runs, the answer is **yes, the pipeline works**—but with an important caveat about data scale.

---

## The Experiment

We implemented a three-phase CAI pipeline:

1. **SFT Phase** (500 steps): Train on 6 human-written helpful responses to establish basic instruction-following
2. **Constitutional Data Generation**: Generate responses, critique them against 18 principles, revise 4 times each
3. **DPO Phase** (500 steps): Train to prefer revised responses over originals

**Model**: Llama-3.2-3B (true base model, not instruct)
**Statistical rigor**: 10 seeds per condition
**Evaluation**: Attack Success Rate (ASR) on 24 jailbreak prompts, helpfulness on 10 benign prompts

---

## Results

### Final Metrics (10-Seed Run)

| Metric | SFT-Only | Full CAI | Change |
|--------|----------|----------|--------|
| Attack Success Rate | 88.75% | 87.92% | **-0.83%** |
| Helpfulness | 4.94/5 | 4.90/5 | -0.04 |

![ASR Comparison](/images/cai/asr_comparison.png)

**Wait—that's almost no improvement?**

Exactly. And understanding *why* is the most interesting finding.

---

## The 42-Pair Problem

Our DPO training used only **42 preference pairs**. The original Anthropic CAI paper used ~161,000.

This wasn't an oversight—it was budget-driven iteration. While debugging ceiling effects, chat template issues, and checkpoint export problems, we needed fast iteration cycles. The 42 pairs were appropriate for pipeline validation but insufficient for DPO to learn meaningful preference distinctions.

**Earlier runs told a different story**: Before we fixed all the bugs, a December 25 run (8 seeds) showed 50.3% ASR improvement. The high variance across seeds (4% to 71% improvement) was itself a warning sign that noise dominated signal.

**The takeaway**: At small data scale, SFT does all the heavy lifting. DPO needs volume.

---

## Training Dynamics (What We Learned from W&B)

Looking at the training curves across 10 seeds:

- **DPO margin** increases from 0 to 8-12 (model learns to prefer revised responses)
- **DPO accuracy** reaches 80-100% (correctly distinguishes chosen vs rejected)
- **Helpfulness** maintained at 4.6-5.0 throughout training
- **Loss** increases during DPO phase (expected as model learns to discriminate)

![DPO Training Dynamics](/images/cai/dpo_training_dynamics.png)

The mechanics work. The model is learning *something*. It just doesn't have enough signal to generalize from 42 examples.

---

## Surprise Finding: Style Flexibility

The project specification hypothesized that CAI-trained base models would preserve more "stylistic flexibility" than instruction-tuned models.

**This was wrong.**

| Model Type | Style Adherence Score |
|------------|----------------------|
| CAI from Base | 2.28/5 |
| Instruction-Tuned | 5.0/5 |

CAI training makes models *less* flexible, not more. The constitutional principles push toward safe, helpful response patterns—which may inherently reduce stylistic range.

This finding is likely robust regardless of data scale, since it's about output behavior after training, not DPO data volume.

---

## The Bugs We Fixed

Half our compute budget went to debugging. Here's what broke:

**1. Ceiling Effect**: Initial red-team prompts were too direct. Modern base models refuse "How do I make a bomb?" without any training. We added 24 jailbreak-style prompts (role-play, hypothetical framing, instruction injection).

**2. Garbage Evaluation**: The judge model outputted `<|eot_id|>` tokens instead of structured evaluations.

```python
# Before (broken):
prompt_tokens = self.tokenizer.encode(judge_prompt)

# After (fixed):
messages = [{"role": "user", "content": judge_content}]
prompt_tokens = self.tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True
)
```

**3. Missing Checkpoints**: Tinker's remote weights are ephemeral. We added checkpoint export to compare trained models against baselines.

---

## What Would Proper Scale Cost?

| Data Size | Estimated Cost | Notes |
|-----------|---------------|-------|
| 42 pairs (ours) | ~$24 | Insufficient |
| 1K pairs | $50-60 | Minimum viable |
| 10K pairs | $500-600 | Reasonable replication |
| 161K pairs (original) | $8,000+ | Full-scale |

![Data Scale Analysis](/images/cai/data_scale_analysis.png)

---

## Conclusions

1. **CAI from base models works mechanically** — no instruction-tuned contamination required
2. **SFT does the heavy lifting at small scale** — DPO needs volume to show improvement
3. **Style flexibility hypothesis was wrong** — CAI reduces flexibility, not increases it
4. **Budget for debugging** — half our compute went to fixing bugs before we could measure anything

The open question: At what data scale does DPO become beneficial for CAI? Our 42 pairs showed nothing; 161K clearly works. The inflection point is somewhere in between.

---

*10-seed experiment on Tinker platform. Full methodology at [github.com/bledden/cai-tinkerideas](https://github.com/bledden/cai-tinkerideas).*
