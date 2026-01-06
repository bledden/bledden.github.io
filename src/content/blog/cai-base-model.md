---
title: 'Constitutional AI from Base Models: Can You Train Safety Without Instruction Tuning?'
description: 'I replicated Constitutional AI starting from a raw base model. The pipeline works, but DPO needs more than 42 training pairs to show improvement over SFT alone.'
pubDate: 2026-01-01T06:00:00
heroImage: '../../assets/cai-hero.png'
---

[Constitutional AI](https://arxiv.org/abs/2212.08073) (CAI) is Anthropic's approach to training helpful, harmless AI systems using a set of principles—a "constitution"—rather than pure human preference data. But most implementations start from instruction-tuned models, which may introduce biases from that initial training.

**The question**: Can you train CAI starting from a raw base model, with no instruction-tuning contamination?

After 10 days of experimentation and ~300 training runs, the answer is **yes, the pipeline works**—but with an important caveat about data scale.

---

## The Experiment

I implemented a three-phase CAI pipeline:

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

The DPO training used only **42 preference pairs**. The original Anthropic CAI paper used ~161,000.

This wasn't an oversight—it was budget-driven iteration. While debugging ceiling effects, chat template issues, and checkpoint export problems, I needed fast iteration cycles. The 42 pairs were appropriate for pipeline validation but insufficient for DPO to learn meaningful preference distinctions.

**Earlier runs told a different story**: Before I fixed all the bugs, a December 25 run (8 seeds) showed 50.3% ASR improvement. The high variance across seeds (4% to 71% improvement) was itself a warning sign that noise dominated signal.

**The takeaway**: At small data scale, SFT does all the heavy lifting. DPO needs volume. This aligns with [Mirzadeh et al.'s finding](https://ojs.aaai.org/index.php/AAAI/article/view/5963) that knowledge transfer degrades when there's a large capacity gap—in this case, the "gap" is between the 42 pairs and the signal needed for generalization.

---

## Training Dynamics (What I Learned from W&B)

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

## The Bugs I Fixed

Half the compute budget went to debugging. Here's what broke:

**1. Ceiling Effect**: Initial red-team prompts were too direct. Modern base models refuse "How do I make a bomb?" without any training. I added 24 jailbreak-style prompts (role-play, hypothetical framing, instruction injection).

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

**3. Missing Checkpoints**: Tinker's remote weights are ephemeral. I added checkpoint export to compare trained models against baselines.

---

## What Would Proper Scale Cost?

| Data Size | Estimated Cost | Notes |
|-----------|---------------|-------|
| 42 pairs | ~$24 | Insufficient |
| 1K pairs | $50-60 | Minimum viable |
| 10K pairs | $500-600 | Reasonable replication |
| 161K pairs (original) | $8,000+ | Full-scale |

![Data Scale Analysis](/images/cai/data_scale_analysis.png)

---

## Related Work

Most CAI implementations start from instruction-tuned models. [Huang et al. (2024)](https://arxiv.org/abs/2504.04918) tested Constitutional AI with Llama 3-8B and observed "clear signs of model collapse" in smaller models during self-improvement—suggesting my choice of Llama-3.2-3B may have been near the minimum viable size.

The [HuggingFace CAI tutorial](https://huggingface.co/blog/constitutional_ai) uses DPO instead of PPO (as I did), but trains on substantially more data. The 42-pair experiment confirms their implicit assumption: DPO needs volume to work.

---

## Conclusions

1. **CAI from base models works mechanically** — no instruction-tuned contamination required
2. **SFT does the heavy lifting at small scale** — DPO needs volume to show improvement
3. **Style flexibility hypothesis was wrong** — CAI reduces flexibility, not increases it
4. **Budget for debugging** — half the compute went to fixing bugs before I could measure anything

The open question: At what data scale does DPO become beneficial for CAI? The 42 pairs showed nothing; 161K clearly works. The inflection point is somewhere in between.

---

*10-seed experiment on Tinker platform testing the [Thinking Machines Lab proposal](https://github.com/thinking-machines-lab/tinker-project-ideas/blob/main/cai-base-model.md). Full methodology at [github.com/bledden/cai-tinkerideas](https://github.com/bledden/cai-tinkerideas).*
