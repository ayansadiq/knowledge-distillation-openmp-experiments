# PyTorch → JAX Translation with Knowledge Distillation

This module extends the repository's original knowledge-distillation experiments from **classification** to **sequence-to-sequence source-code translation**.

## Research question

Can a smaller student model learn to translate PyTorch source into semantically equivalent JAX using both:

1. reference JAX code as the supervised target, and
2. a frozen teacher model's token distributions as a distillation signal?

The training objective is:

```text
loss = alpha * cross_entropy(reference_jax, student)
     + (1 - alpha) * T^2 * KL(teacher || student)
```

## Pipeline

```text
PyTorch source ──────┬────> frozen teacher ────> teacher token distribution
                     │
                     └────> student ───────────> student JAX tokens
                                      ▲
                                      │
                               reference JAX
```

## Upstream benchmark sources

- PyTorch translations: https://github.com/tuanng04/jaxbench-pytorch-translation
- JAX references: https://github.com/AI-Hypercomputer/accelerator-agents/tree/main/JAXBench/benchmark

The first real case is **10p_Sparse_MoE**.

## Why text-to-text first?

The previous baseline predicts classes. Code translation requires generating an entire token sequence. The first milestone therefore verifies that the same teacher/student KD idea works when **both the input and output are text**.

## Status

- [x] define sequence-to-sequence KD objective
- [x] identify corresponding Sparse MoE PyTorch/JAX pair
- [ ] text-to-text KD proof
- [ ] automatic paired benchmark builder
- [ ] PyTorch→JAX training harness
- [ ] generated-code evaluation
- [ ] functional equivalence tests

## Important limitation

The initial FLAN-T5 setup is a **mechanism-validation baseline**, not a claim that full benchmark files can already be translated accurately. Many source files exceed its useful context length. The next research step is function-level alignment or a longer-context code model.
