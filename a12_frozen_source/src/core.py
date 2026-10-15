from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def tikhonov(R,y,lam=1e-3,clamp=True):
    B,M,N=R.shape; A=R/N
    G=torch.einsum("bmn,bkn->bmk",A,A)
    I=torch.eye(M,device=R.device,dtype=R.dtype).expand(B,M,M)
    alpha=torch.linalg.solve(G+lam*I,y.unsqueeze(-1)).squeeze(-1)
    x=torch.einsum("bmn,bm->bn",A,alpha)
    return x.clamp_min(0.0) if clamp else x

def rel_l2(p,x):
    return torch.linalg.vector_norm(p-x,dim=-1)/torch.linalg.vector_norm(x,dim=-1).clamp_min(1e-8)
def sam(p,x):
    c=(p*x).sum(-1)/(torch.linalg.vector_norm(p,dim=-1)*torch.linalg.vector_norm(x,dim=-1)).clamp_min(1e-8)
    return torch.acos(c.clamp(-1,1))
def forward_residual(R,x,y):
    A=R/x.shape[-1];yp=torch.einsum("bmn,bn->bm",A,x)
    return torch.linalg.vector_norm(yp-y,dim=-1)/torch.linalg.vector_norm(y,dim=-1).clamp_min(1e-8)

class ResBlock(nn.Module):
    def __init__(self,w,d):
        super().__init__()
        self.c1=nn.Conv1d(w,w,5,padding=2*d,dilation=d)
        self.c2=nn.Conv1d(w,w,5,padding=2*d,dilation=d)
        self.n1=nn.GroupNorm(8,w);self.n2=nn.GroupNorm(8,w)
    def forward(self,x):
        h=F.gelu(self.n1(self.c1(x)));h=self.n2(self.c2(h))
        return F.gelu(x+0.25*h)

class FiRNO(nn.Module):
    """Fisher-Gated Range-Null Operator.

    raw mode: learned residual, no range-null gate.
    hard_null: only P_null(delta).
    fisher_gate: learned per-singular-mode gate using Fisher strength,
                 residual mode energy and correction mode energy.
    Optional data-proximal correction softly returns the solution toward
    measurement fidelity, appropriate for noisy observations.
    """
    def __init__(self,width=64,mode="fisher_gate",prox=True,use_fisher_channel=True):
        super().__init__()
        self.mode=mode;self.prox=prox;self.use_fisher_channel=use_fisher_channel
        self.inp=nn.Conv1d(4 if use_fisher_channel else 3,width,7,padding=3)
        self.blocks=nn.ModuleList([ResBlock(width,d) for d in [1,2,4,8,4,2,1]])
        self.out=nn.Conv1d(width,1,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
        self.step=nn.Parameter(torch.tensor(-1.15))
        self.mode_gate=nn.Sequential(nn.Linear(3,24),nn.GELU(),nn.Linear(24,1))
        nn.init.constant_(self.mode_gate[-1].bias,1.5) # start close to raw residual
        self.prox_gate=nn.Sequential(nn.Linear(5,24),nn.GELU(),nn.Linear(24,1))
        nn.init.constant_(self.prox_gate[-1].bias,-1.1)

    @staticmethod
    def fisher_map(R,noise_std):
        A=R/R.shape[-1]
        f=(A*A).sum(1)/(float(noise_std)**2+1e-10)
        z=torch.log1p(f)
        return (z-z.mean(-1,keepdim=True))/(z.std(-1,keepdim=True)+1e-6)

    def split_and_gate(self,R,y,x0,d,noise_std):
        A=R/R.shape[-1]
        # Thin SVD: row-space basis is Vh, one mode per nonzero singular direction.
        U,S,Vh=torch.linalg.svd(A,full_matrices=False)
        c=torch.einsum("bmn,bn->bm",Vh,d)
        drow=torch.einsum("bmn,bm->bn",Vh,c)
        dnull=d-drow
        if self.mode=="raw":
            gates=torch.ones_like(S)
            return d,gates,S,dnull,drow
        if self.mode=="hard_null":
            gates=torch.zeros_like(S)
            return dnull,gates,S,dnull,drow
        # Fisher strength in each singular mode.
        lf=torch.log(S*S/(float(noise_std)**2+1e-10)+1e-10)
        lf=(lf-lf.mean(-1,keepdim=True))/(lf.std(-1,keepdim=True)+1e-6)
        # Residual coordinates and proposed correction coordinates.
        resid=y-torch.einsum("bmn,bn->bm",A,x0)
        ur=torch.einsum("bmi,bm->bi",U,resid)
        ur=torch.abs(ur)/(torch.abs(ur).mean(-1,keepdim=True)+1e-6)
        cn=torch.abs(c)/(torch.abs(c).mean(-1,keepdim=True)+1e-6)
        feat=torch.stack([lf,ur,cn],-1)
        gates=torch.sigmoid(self.mode_gate(feat).squeeze(-1))
        dg=torch.einsum("bmn,bm->bn",Vh,gates*c)
        return dnull+dg,gates,S,dnull,drow

    def proximal(self,z,R,y,S,gates,noise_std):
        A=R/R.shape[-1]
        resid=y-torch.einsum("bmn,bn->bm",A,z)
        G=torch.einsum("bmn,bkn->bmk",A,A)
        I=torch.eye(G.shape[-1],device=G.device,dtype=G.dtype).expand_as(G)
        # instrument identifiability signature
        logS=torch.log(S+1e-8)
        cond=logS.max(-1).values-logS.min(-1).values
        feats=torch.stack([
            logS.mean(-1),logS.std(-1),cond,
            gates.mean(-1),
            torch.linalg.vector_norm(resid,dim=-1)/(torch.linalg.vector_norm(y,dim=-1)+1e-8)
        ],-1)
        eta=torch.sigmoid(self.prox_gate(feats)).squeeze(-1)
        ridge=max(float(noise_std)**2,1e-6)
        alpha=torch.linalg.solve(G+ridge*I,resid.unsqueeze(-1)).squeeze(-1)
        corr=torch.einsum("bmn,bm->bn",A,alpha)
        return z+eta[:,None]*corr,eta

    def forward(self,R,y,noise_std=0.01):
        x0=tikhonov(R,y,1e-3)
        A=R/R.shape[-1]
        bp=torch.einsum("bmn,bm->bn",A,y)
        bp=bp/(bp.abs().amax(-1,keepdim=True)+1e-6)
        grad=torch.zeros_like(x0);grad[:,1:-1]=0.5*(x0[:,2:]-x0[:,:-2])
        feats=[x0,bp,grad]
        fmap=self.fisher_map(R,noise_std)
        if self.use_fisher_channel:feats.append(fmap)
        h=self.inp(torch.stack(feats,1))
        for b in self.blocks:h=b(h)
        raw=torch.tanh(self.out(h).squeeze(1))
        raw=(0.60*torch.sigmoid(self.step))*raw
        d,gates,S,dnull,drow=self.split_and_gate(R,y,x0,raw,noise_std)
        z=x0+d
        eta=torch.zeros(R.shape[0],device=R.device)
        if self.prox:
            z,eta=self.proximal(z,R,y,S,gates,noise_std)
        x=z.clamp_min(0.0)
        return {"x":x,"x0":x0,"gates":gates,"singular_values":S,"eta":eta,
                "raw":raw,"dnull":dnull,"drow":drow,"fisher_map":fmap}
