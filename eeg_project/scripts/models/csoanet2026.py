#CSOANet2026+CSOA: stride temporal + residual + cross-scale attention
import torch,torch.nn as nn,torch.nn.functional as F
def auto_cfg(c,t,n=2):
 F1,D,d=(8,2,.3)if n<=2 else(16,2,.35)if n<=5 else(16,4,.4)if n<=12 else(32,4,.45)
 kt=max(t//50,5)|1;t2=t//2;p1=min(4,max(2,t2//64));t3=t2//p1
 k=max(t3//30,3)|1;d1=max(t3//80,2);d2=d1*4
 return dict(ch=c,t=t,nc=n,F1=F1,D=D,kt=kt,k=k,d1=d1,d2=d2,p1=p1,p2=min(4,max(2,t3//16)),dr=d)
TASK=auto_cfg(62,2000,2)
class CSOANet2026(nn.Module):
 def __init__(s,x=None,nc=2,**kw):
  super().__init__()
  if isinstance(x,torch.Tensor):c=auto_cfg(x.shape[-2],x.shape[-1],nc)
  elif isinstance(x,dict):c=dict(x)
  else:c=dict(TASK)
  c.update(kw);s.c=c;F2=c['F1']*c['D'];k=c['k'];kt=c['kt']
  #Stage1: stride temporal(RF=kt, halves T) + spatial
  s.s1=nn.Sequential(
   nn.Conv2d(1,c['F1'],(1,kt),stride=(1,2),padding=(0,(kt-1)//2),bias=0),nn.BatchNorm2d(c['F1']),nn.ELU(1),
   nn.Conv2d(c['F1'],F2,(c['ch'],1),groups=c['F1'],bias=0),nn.BatchNorm2d(F2),
   nn.ELU(1),nn.AvgPool2d((1,c['p1'])),nn.Dropout(c['dr']))
  #Stage2: CSOA with residual
  s.d0=nn.Conv2d(F2,F2,(1,k),padding=(0,(k-1)//2),groups=F2,bias=0)
  s.d1=nn.Conv2d(F2,F2,(1,k),dilation=(1,c['d1']),padding=(0,c['d1']*(k-1)//2),groups=F2,bias=0)
  s.d2=nn.Conv2d(F2,F2,(1,k),dilation=(1,c['d2']),padding=(0,c['d2']*(k-1)//2),groups=F2,bias=0)
  s.csoa=nn.Linear(6,6)
  s.bn2=nn.BatchNorm2d(F2)
  #Stage3: sep + head
  s.b3=nn.Sequential(nn.Conv2d(F2,F2,(1,k),padding=(0,(k-1)//2),groups=F2,bias=0),nn.Conv2d(F2,F2,(1,1),bias=0),nn.BatchNorm2d(F2),nn.ELU(1),nn.AvgPool2d((1,c['p2'])),nn.Dropout(c['dr']))
  s.da=nn.Parameter(torch.ones(1,F2,1,1));s.po=nn.AdaptiveAvgPool2d(1);s.fc=nn.Linear(F2,c['nc'])
 def forward(s,x):
  x=s.s1(x)
  res=x
  x0,x1,x2=s.d0(x),s.d1(x),s.d2(x)
  c01,c12,c02=x0*x1,x1*x2,x0*x2
  terms=[x0,x1,x2,c01,c12,c02]
  g=torch.stack([t.mean([1,2,3])for t in terms],1)
  w=F.softmax(s.csoa(g),1)
  x=sum(w[:,i,None,None,None]*terms[i]for i in range(6))
  x=s.b3(F.elu(s.bn2(x+res)))
  return s.fc(s.po(x*torch.sigmoid(s.da)).flatten(1))
 @staticmethod
 def max_norm(m,v=.25):
  for n,p in m.named_parameters():
   if'weight'in n and p.dim()>=2:
    with torch.no_grad():nr=p.norm(2,dim=tuple(range(1,p.dim())),keepdim=True);p.mul_(torch.clamp(nr,max=v)/(nr+1e-8))
