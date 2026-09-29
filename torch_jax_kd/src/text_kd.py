"""Minimal text-to-text knowledge-distillation proof of concept."""
import argparse, random, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PAIRS=[(f"describe parity: {n}",f"{n} is {'even' if n%2==0 else 'odd'}") for n in range(2,12)]
class Pairs(Dataset):
    def __len__(self): return len(PAIRS)
    def __getitem__(self,i): return PAIRS[i]

def collate(batch,tok):
    x,y=zip(*batch)
    enc=tok(list(x),padding=True,truncation=True,max_length=64,return_tensors="pt")
    tgt=tok(text_target=list(y),padding=True,truncation=True,max_length=64,return_tensors="pt")
    labels=tgt.input_ids; labels[labels==tok.pad_token_id]=-100
    return enc,labels

def kd_loss(s,t,labels,T):
    kl=F.kl_div(F.log_softmax(s/T,-1),F.softmax(t/T,-1),reduction="none").sum(-1)
    mask=labels.ne(-100)
    return (kl*mask).sum()/mask.sum().clamp_min(1)*T*T

def main(a):
    random.seed(42); torch.manual_seed(42)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=AutoTokenizer.from_pretrained(a.teacher)
    teacher=AutoModelForSeq2SeqLM.from_pretrained(a.teacher).to(dev).eval()
    student=AutoModelForSeq2SeqLM.from_pretrained(a.student).to(dev)
    for p in teacher.parameters(): p.requires_grad=False
    loader=DataLoader(Pairs(),batch_size=2,shuffle=True,collate_fn=lambda b:collate(b,tok))
    opt=torch.optim.AdamW(student.parameters(),lr=5e-5)
    for epoch in range(a.epochs):
        student.train()
        for x,labels in loader:
            x={k:v.to(dev) for k,v in x.items()}; labels=labels.to(dev)
            with torch.no_grad(): t=teacher(**x,labels=labels)
            s=student(**x,labels=labels); kd=kd_loss(s.logits,t.logits,labels,a.temperature)
            loss=.5*s.loss+.5*kd; opt.zero_grad(); loss.backward(); opt.step()
        print(f"epoch={epoch+1} loss={loss.item():.4f}")
    student.eval()
    for prompt in ["describe parity: 12","describe parity: 17"]:
        x=tok(prompt,return_tensors="pt").to(dev); out=student.generate(**x,max_new_tokens=12)
        print(prompt,"->",tok.decode(out[0],skip_special_tokens=True))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--teacher",default="google/flan-t5-base")
    p.add_argument("--student",default="google/flan-t5-small"); p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--temperature",type=float,default=2.0); main(p.parse_args())
