"""Lightweight generated-code diagnostics."""
import argparse, ast, difflib
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("generated",type=Path); p.add_argument("reference",type=Path)
a=p.parse_args(); g=a.generated.read_text(); r=a.reference.read_text()
try: ast.parse(g); syntax=True
except SyntaxError: syntax=False
print({"python_syntax_valid":syntax,"character_similarity":round(difflib.SequenceMatcher(None,g,r).ratio(),4)})
print("Lexical similarity is diagnostic only; functional equivalence is the real target.")
