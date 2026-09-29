"""Minimal text-to-text knowledge-distillation proof of concept.

Purpose: verify that the existing KD idea works when both input and output are
token sequences, before applying it to PyTorch -> JAX code translation.
"""
import argparse
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


PAIRS = [
    (f"describe parity: {n}", f"{n} is {'even' if n % 2 == 0 else 'odd'}")
    for n in range(2, 12)
]


class PairDataset(Dataset):
    def __len__(self):
        return len(PAIRS)

    def __getitem__(self, index):
        return PAIRS[index]


def collate(batch, tokenizer):
    inputs, targets = zip(*batch)
    encoded = tokenizer(
        list(inputs),
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    target = tokenizer(
        text_target=list(targets),
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    labels = target.input_ids
    labels[labels == tokenizer.pad_token_id] = -100
    return encoded, labels


def masked_kd_loss(student_logits, teacher_logits, labels, temperature):
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
    parser.add_argument("--teacher", default="google/flan-t5-base")
    parser.add_argument("--student", default="google/flan-t5-small")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForSeq2SeqLM.from_pretrained(args.teacher).to(device).eval()
    student = AutoModelForSeq2SeqLM.from_pretrained(args.student).to(device)

    for parameter in teacher.parameters():
        parameter.requires_grad = False

    loader = DataLoader(
        PairDataset(),
        batch_size=2,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer),
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0

        for inputs, labels in loader:
            inputs = {key: value.to(device) for key, value in inputs.items()}
            labels = labels.to(device)

            with torch.no_grad():
                teacher_output = teacher(**inputs, labels=labels)

            student_output = student(**inputs, labels=labels)

            kd_loss = masked_kd_loss(
                student_output.logits,
                teacher_output.logits,
                labels,
                args.temperature,
            )
            loss = (
                args.alpha * student_output.loss
                + (1.0 - args.alpha) * kd_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"epoch={epoch + 1} loss={total_loss / len(loader):.4f}")

    student.eval()
    for prompt in ["describe parity: 12", "describe parity: 17"]:
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        output = student.generate(**encoded, max_new_tokens=12)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"{prompt} -> {decoded}")


if __name__ == "__main__":
    main()
