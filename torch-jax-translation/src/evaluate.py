"""Lightweight diagnostics for generated JAX source."""

import argparse
import ast
import difflib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("generated", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()

    generated = args.generated.read_text()
    reference = args.reference.read_text()

    try:
        ast.parse(generated)
        syntax_valid = True
    except SyntaxError:
        syntax_valid = False

    similarity = difflib.SequenceMatcher(
        None,
        generated,
        reference,
    ).ratio()

    print(
        {
            "python_syntax_valid": syntax_valid,
            "character_similarity": round(similarity, 4),
        }
    )
    print(
        "Lexical similarity is only a diagnostic; "
        "functional equivalence is the research target."
    )


if __name__ == "__main__":
    main()
