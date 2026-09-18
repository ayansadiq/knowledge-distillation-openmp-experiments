# PyTorch Knowledge Distillation Baseline

This repository reproduces a knowledge-distillation experiment using
PyTorch and the CIFAR-10 image-classification dataset.

## Purpose

The goal is to understand how a smaller student model can learn from a
larger teacher model without increasing the student's parameter count
or inference cost.

## Models Compared

1. A larger teacher model
2. A smaller student trained normally using hard labels
3. The same student architecture trained using hard labels and the
   teacher's softened probability outputs

## Knowledge-Distillation Objective

The student is trained using two losses:

- Hard-label loss: compares the student with the correct class label
- Soft-target loss: compares the student's probability distribution
  with the teacher's softened probability distribution

Temperature controls how soft the teacher and student distributions are.

## Baseline Settings

- Dataset: CIFAR-10
- Framework: PyTorch
- Temperature: 2
- Soft-target loss weight: 0.25
- Cross-entropy loss weight: 0.75
- Student and teacher optimization: Adam

## Results

| Model | Accuracy |
|---|---:|
| Teacher | Pending |
| Student without distillation | Pending |
| Student with distillation | Pending |

## Source-Code Experiment

After reproducing the baseline, the temperature is changed to observe
how the softness of the teacher's output affects student performance.

## Connection to OpenMP Code Generation

This image-classification experiment serves as a small, understandable
knowledge-distillation baseline. A future extension could use a large
teacher LLM or agent system to generate OpenMP C/C++ programs.

Generated programs would be compiled, tested for correctness, and
benchmarked. Verified sequential/OpenMP program pairs could then be
used to fine-tune a smaller code-generation model.

## Source

Based on the official PyTorch Knowledge Distillation Tutorial:

https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
