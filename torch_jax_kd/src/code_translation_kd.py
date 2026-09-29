"""KD training harness for PyTorch-source -> JAX-source translation."""
import argparse,json,torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer

class Pairs(Dataset):
    def __init__(self,path): self.rows=[json.loads(x) for x in open(path) if x.strip()]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return self.rows[i]

def collate(rows,tok,ms,mt):
    x=tok(["Translate this PyTorch implementation to semantically equivalent JAX:\n"+r["input"] for r in rows],
          padding=True,truncation=True,max_length=ms,return_tensors="pt")
    y=tok(text_target=[r["target"] for r in rows],padding=True,truncation=True,max_length=mt,return_tensors="pt")
    labels=y.input_ids; labels[labels==tok.pad_token_id]=-100
    return x,labels

def kd_loss(s,t,labels,T):
    kl=F.kl_div(F.log_softmax(s/T,-1),F.softmax(t/T,-1),reduction="none").sum(-1)
    mask=labels.ne(-100); return (kl*mask).sum()/mask.sum().clamp_min(1)*T*T

def main(a):
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=AutoTokenizer.from_pretrained(a.teacher)
    teacher=AutoModelForSeq2SeqLM.from_pretrained(a.teacher).to(dev).eval()
    student=AutoModelForSeq2SeqLM.from_pretrained(a.student).to(dev)
    for p in teacher.parameters(): p.requires_grad=False
    loader=DataLoader(Pairs(a.data),batch_size=1,shuffle=True,collate_fn=lambda r:collate(r,tok,a.max_source,a.max_target))
    opt=torch.optim.AdamW(student.parameters(),lr=5e-5)
    for epoch in range(a.epochs):
        student.train()
        for step,(x,labels) in enumerate(loader,1):
            x={k:v.to(dev) for k,v in x.items()}; labels=labels.to(dev)
            with torch.no_grad(): t=teacher(**x,labels=labels)
            s=student(**x,labels=labels); kd=kd_loss(s.logits,t.logits,labels,2.0)
            loss=.5*s.loss+.5*kd; opt.zero_grad(); loss.backward(); opt.step()
            print(f"epoch={epoch+1} step={step} ce={s.loss.item():.4f} kd={kd.item():.4f} total={loss.item():.4f}")
    student.save_pretrained(a.output); tok.save_pretrained(a.output)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data",default="data/pairs.jsonl")
    p.add_argument("--teacher",default="google/flan-t5-base"); p.add_argument("--student",default="google/flan-t5-small")
    p.add_argument("--epochs",type=int,default=1); p.add_argument("--max-source",type=int,default=512)
    p.add_argument("--max-target",type=int,default=512); p.add_argument("--output",default="checkpoints/student")
    main(p.parse_args())
