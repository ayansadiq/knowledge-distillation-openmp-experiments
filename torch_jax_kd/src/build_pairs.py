"""Build aligned PyTorch -> JAX source-code pairs from two checkouts."""
import argparse, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--torch-root", required=True, type=Path)
    p.add_argument("--jax-root", required=True, type=Path)
    p.add_argument("--output", default="data/pairs.jsonl", type=Path)
    a=p.parse_args(); rows=[]
    for src in a.torch_root.glob("*/baseline.py"):
        name=src.parent.name
        matches=list(a.jax_root.glob(f"level*/{name}/baseline.py"))
        if len(matches)==1:
            rows.append({"benchmark":name,"source_framework":"pytorch","target_framework":"jax",
                         "input":src.read_text(),"target":matches[0].read_text(),
                         "torch_path":str(src),"jax_path":str(matches[0])})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w") as f:
        for row in rows: f.write(json.dumps(row)+"\n")
    print(f"wrote {len(rows)} aligned pairs to {a.output}")

if __name__=="__main__": main()
