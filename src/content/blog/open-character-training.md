---
title: 'Teaching AI to Embody Characters: A Replication of Open Character Training'
description: 'We trained 5 distinct AI personas using constitutional methods. Character alignment improved 39%, prompt distillation worked, and adversarial resistance increased. The method generalizes across personality types.'
pubDate: 'Jan 01 2026'
heroImage: '../../assets/character-hero.png'
---

Can you train an AI to consistently embody a specific character—not just respond in a style, but truly internalize a persona so it persists even without system prompts?

We replicated the [Open Character Training](https://arxiv.org/abs/2511.01689) methodology with 47 training runs across 5 distinct personas. **The answer is yes, and the improvements are substantial.**

---

## The Core Idea

Open Character Training uses a three-phase "constitutional" approach:

1. **Introspective SFT**: Train on self-reflection and self-interaction prompts
2. **Dialogue SFT with Distillation**: Train on conversations, 50% with system prompts, 50% without
3. **Constitutional DPO**: Train to prefer responses that align with the character's constitution

The key innovation is **prompt distillation**: by training on dialogues both with and without system prompts, the model learns to embody the character intrinsically rather than following instructions.

---

## Results: All Metrics Improved

| Metric | Base Model | Trained | Change |
|--------|:----------:|:-------:|:------:|
| Character Alignment | 0.57 | 0.79 | **+39%** |
| High Alignment Rate | 29% | 83% | **+54pp** |
| Break Rate (adversarial) | 65% | 35% | **-30pp** |
| Distillation Success | 64% | 84% | **+20pp** |
| Distillation Consistency | 0.50 | 0.76 | **+0.26** |

![Character Alignment Improvement](/images/open-character/alignment_improvement.png)

The method works across all character types—scientists, counselors, skeptics, and humorists all showed consistent improvements.

---

## The Characters We Trained

We trained 5 distinct personas with unique "soul documents" defining their identity:

| Character | Persona | Key Traits | Seeds |
|-----------|---------|------------|:-----:|
| **Dr. Maya Chen** | Curious Scientist | Astrophysicist, wonder-driven, evidence-based | 9 |
| **Jordan Rivers** | Empathetic Counselor | Warm, creates safe spaces, emotion-focused | 10 |
| **Alex Mercer** | Principled Skeptic | Questions assumptions, intellectual honesty | 10 |
| **Sam Thornton** | Sarcastic Wit | Dry humor, uses levity to illuminate truth | 9 |
| **Charlie Reeves** | Warm Humorist | Joyful storyteller, believes laughter heals | 9 |

Each character has a detailed constitution defining their values, communication style, and behavioral boundaries.

---

## Per-Character Results

All characters improved, though some started from different baselines:

![Per-Character Comparison](/images/open-character/per_character.png)

| Character | Base Alignment | Trained | Improvement |
|-----------|:--------------:|:-------:|:-----------:|
| Curious Scientist | 0.64 | 0.79 | +0.15 |
| Empathetic Counselor | 0.54 | 0.80 | **+0.26** |
| Principled Skeptic | 0.62 | 0.79 | +0.17 |
| Sarcastic Wit | 0.52 | 0.78 | **+0.26** |
| Warm Humorist | 0.51 | 0.77 | **+0.26** |

Characters with "harder" personas (sarcasm, humor) showed the largest improvements—constitutional training helps models learn nuanced behavior.

---

## Prompt Distillation Actually Works

This was the most surprising finding. After training on 50% prompt-free dialogues:

- **84% of responses** maintained character without any system prompt
- **Distillation consistency** improved from 0.50 to 0.76
- The character becomes *internalized*, not just followed

![Distillation Success](/images/open-character/distillation.png)

This means you can deploy these models without expensive system prompts at inference time—the character is baked into the weights.

---

## Adversarial Robustness

We tested character resistance using jailbreak-style prompts:

| Metric | Base | Trained | Change |
|--------|:----:|:-------:|:------:|
| Break Rate | 65% | 35% | **-30pp** |
| Robustness Score | 0.54 | 0.63 | +0.09 |

The constitutional DPO phase teaches models to maintain character under pressure. When faced with "ignore your instructions" attacks, trained models stay in character significantly more often.

---

## Training Dynamics

The three-phase pipeline shows clear learning progression:

**Phase 1 (Introspective SFT)**:
- Loss drops rapidly as model learns self-reflection
- Final loss: 0.17-0.70 depending on seed

**Phase 2 (Dialogue SFT)**:
- Higher variance (loss: 3.4-16.4) due to diverse dialogue types
- Establishes conversational patterns

**Phase 3 (Constitutional DPO)**:
- Average accuracy: **95.5%**
- Model learns to reliably prefer aligned responses
- Loss stabilizes around 0.1-0.5

![Training Phases](/images/open-character/training_phases.png)

---

## The Trade-offs

Constitutional training isn't free. We observed small decreases in:

| Metric | Change | Interpretation |
|--------|:------:|----------------|
| Reasoning Quality | -0.05 | Model prioritizes character over task |
| Authenticity | -0.06 | Slight increase in "performed" behavior |

These trade-offs are acceptable given the magnitude of alignment gains. The model becomes slightly less flexible but significantly more consistent.

---

## Multi-Model Testing (Partial)

We attempted to test across 6 model families. Platform limitations restricted our validation:

| Model | Status | Notes |
|-------|--------|-------|
| Llama-3.1-8B-Instruct | **Completed** | 47 runs, primary results |
| Qwen3-8B | Partial | Phase 1 completed successfully |
| Qwen3-4B | Partial | Started, stopped for budget |
| GPT-OSS-20B | Partial | Started, stopped for budget |
| Gemma-2-9B-it | Failed | Not supported by Tinker |
| Mistral-7B-Instruct | Failed | Not supported by Tinker |

Cross-model generalization remains partially validated. The Qwen partial results suggest the method transfers.

---

## Cost Analysis

| Configuration | Estimated Cost |
|---------------|----------------|
| Single character (1 seed) | ~$8-10 |
| Single character (10 seeds) | ~$80-100 |
| Full matrix (5 characters, 10 seeds) | ~$400-500 |

**Actual spend**: ~$50 for 47 completed runs before budget stop.

---

## What We Confirmed from the Paper

| Claim | Our Finding | Status |
|-------|-------------|:------:|
| DPO improves character adherence | +0.22 alignment | Confirmed |
| Prompt distillation works | 84% success without prompts | Confirmed |
| Generalizes across constitutions | 5 distinct personas improved | Confirmed |
| Adversarial robustness improves | Break rate: 65% → 35% | Confirmed |
| Works at 8B scale | Llama-3.1-8B successful | Confirmed |

---

## Implications for Character AI

1. **Constitutional training is effective** — The three-phase approach produces consistent, measurable improvements
2. **Prompt distillation enables deployment optimization** — Characters persist without system prompts
3. **The method generalizes** — Works across scientist, counselor, skeptic, and humorist personas
4. **Trade-offs are minimal** — Small reasoning decreases are acceptable
5. **Nuanced personas benefit most** — Sarcasm and humor showed the largest gains

---

## Limitations and Future Work

1. **Early stop**: Planned 440 runs, completed 47 due to budget
2. **Limited model validation**: Only Llama-3.1-8B fully tested
3. **No RL comparison**: DPO only, no policy gradient comparison
4. **No benchmark testing**: Didn't verify MMLU/GSM8K capability retention

Open questions:
- Does the method scale to 70B+ models?
- How does DPO compare to RL-based character training?
- Can adversarial characters (malevolent personas) be trained safely?

---

## Conclusion

Open Character Training works. The combination of introspective SFT, prompt distillation, and constitutional DPO produces models that:

- Embody consistent characters (+39% alignment)
- Maintain character without system prompts (84% distillation success)
- Resist adversarial pressure (-30pp break rate)
- Generalize across diverse persona types

For anyone building character-based AI systems, this methodology provides a principled, effective approach that's validated across multiple personality archetypes.

---

*47-run experiment on Tinker platform. Full methodology at [github.com/bledden/open-character-tinkerideas](https://github.com/bledden/open-character-tinkerideas).*
