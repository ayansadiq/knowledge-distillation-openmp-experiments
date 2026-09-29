"""Sequence-to-sequence KD harness for PyTorch -> JAX source translation."""

import argparse
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class TranslationPairs(Dataset):
    def __init__(self, path):
        with open(path) as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate(rows, tokenizer, max_source, max_target):
    prompts = [
        "Translate this PyTorch implementation to semantically equivalent JAX:\n"
        + row["input"]
        for row in rows
    ]
    targets = [row["target"] for row in rows]

    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_source,
        return_tensors="pt",
    )
    target_tokens = tokenizer(
        text_target=targets,
        padding=True,
        truncation=True,
        max_length=max_target,
        return_tensors="pt",
    )

    labels = target_tokens.input_ids
    labels[labels == tokenizer.pad_token_id] = -100
    return inputs, labels


def kd_loss(student_logits, teacher_logits, labels, temperature):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    token_kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
    ).sum(-1)

    mask = labels.ne(-100)
    return (
        (token_kl * mask).sum() / mask.sum().clamp_min(1)
    ) * temperature**2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/pairs.jsonl")
    parser.add_argument("--teacher", default="google/flan-t5-base")
    parser.add_argument("--student", default="google/flan-t5-small")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-source", type=int, default=512)
    parser.add_argument("--max-target", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output", default="checkpoints/student")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)

    teacher = AutoModelForSeq2SeqLM.from_pretrained(args.teacher).to(device).eval()
    student = AutoModelForSeq2SeqLM.from_pretrained(args.student).to(device)

    for parameter in teacher.parameters():
        parameter.requires_grad = False

    loader = DataLoader(
        TranslationPairs(args.data),
        batch_size=1,
        shuffle=True,
        collate_fn=lambda rows: collate(
            rows,
            tokenizer,
            args.max_source,
            args.max_target,
        ),
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for step, (inputs, labels) in enumerate(loader, start=1):
            inputs = {key: value.to(device) for key, value in inputs.items()}
            labels = labels.to(device)

            with torch.no_grad():
                teacher_output = teacher(**inputs, labels=labels)

            student_output = student(**inputs, labels=labels)

            distillation_loss = kd_loss(
                student_output.logits,
                teacher_output.logits,
                labels,
                args.temperature,
            )

            loss = (
                args.alpha * student_output.loss
                + (1.0 - args.alpha) * distillation_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f"epoch={epoch + 1} step={step} "
                f"ce={student_output.loss.item():.4f} "
                f"kd={distillation_loss.item():.4f} "
                f"total={loss.item():.4f}"
            )

    student.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
