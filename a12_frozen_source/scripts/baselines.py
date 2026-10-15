from __future__ import annotations
import math, random
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from src.core import tikhonov, rel_l2, sam, forward_residual
from src.synthetic_v2 import random_spectra, make_wavelength_grid, measure


def prox_disp(R,y,z,ridge=1e-3):
    N=z.shape[-1]; A=R/N
    if A.ndim==2: A=A.unsqueeze(0).expand(z.shape[0],-1,-1)
    resid=y-torch.einsum('bmn,bn->bm',A,z)
    G=torch.einsum('bmn,bkn->bmk',A,A)
    I=torch.eye(G.shape[-1],device=z.device,dtype=z.dtype).expand_as(G)
    alpha=torch.linalg.solve(G+ridge*I,resid.unsqueeze(-1)).squeeze(-1)
    return torch.einsum('bmn,bm->bn',A,alpha)


def nnls_tv_batch(R,y,lam=1e-3,iters=120,rho=1.0):
    """Batched nonnegative TV ADMM for one fixed operator R[M,N]."""
    B=y.shape[0]; M,N=R.shape; A=R/N
    x=tikhonov(R.unsqueeze(0).expand(B,-1,-1),y,1e-3)
    D=torch.zeros(N-1,N,device=y.device,dtype=y.dtype); idx=torch.arange(N-1,device=y.device); D[idx,idx]=-1;D[idx,idx+1]=1
    z=x[:,1:]-x[:,:-1]; u=torch.zeros_like(z)
    H=A.T@A+rho*(D.T@D)+1e-6*torch.eye(N,device=y.device,dtype=y.dtype)
    L=torch.linalg.cholesky(H); b0=y@A
    for _ in range(iters):
        rhs=b0+rho*(z-u)@D
        x=torch.cholesky_solve(rhs.T,L).T.clamp_min(0)
        dx=x[:,1:]-x[:,:-1]; v=dx+u
        z=torch.sign(v)*torch.relu(torch.abs(v)-lam/rho); u=u+dx-z
    return x


class Denoiser1D(nn.Module):
    def __init__(self,width=88):
        super().__init__()
        self.net=nn.Sequential(
          nn.Conv1d(1,width,5,padding=2),nn.GELU(),
          nn.Conv1d(width,width,5,padding=2),nn.GELU(),
          nn.Conv1d(width,width,5,padding=2),nn.GELU(),
          nn.Conv1d(width,width,5,padding=2),nn.GELU(),
          nn.Conv1d(width,1,5,padding=2))
    def forward(self,x): return (x+self.net(x[:,None]).squeeze(1)).clamp_min(0)


def train_pnp_denoiser(out,device,quick=False):
    ck=Path(out)/'PNP_DENOISER.pt'
    if ck.exists():
        d=torch.load(ck,map_location=device,weights_only=False);m=Denoiser1D().to(device);m.load_state_dict(d['model']);m.eval();return m,d
    steps=4 if quick else 700; torch.manual_seed(17001); random.seed(17001)
    wl=make_wavelength_grid(256,device=device);m=Denoiser1D().to(device);opt=torch.optim.AdamW(m.parameters(),lr=7e-4,weight_decay=1e-5)
    # fixed denoising validation
    st=torch.random.get_rng_state();torch.manual_seed(17101);vx=random_spectra(256 if not quick else 8,wl,family='random');vn=0.04*torch.randn_like(vx);vy=(vx+vn).clamp_min(0);torch.random.set_rng_state(st)
    best=1e9;beststep=0
    for step in range(1,steps+1):
        x=random_spectra(64 if not quick else 4,wl,family='random');sig=float(torch.empty(1).uniform_(0.005,0.08)); noisy=(x+sig*torch.randn_like(x)).clamp_min(0)
        p=m(noisy);loss=((p-x)**2).mean();opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
        if step%100==0 or step==steps:
            with torch.no_grad():v=float(((m(vy)-vx)**2).mean())
            if v<best: best=v;beststep=step;torch.save({'model':m.state_dict(),'best_val_mse':best,'step':step},ck)
    d=torch.load(ck,map_location=device,weights_only=False);m.load_state_dict(d['model']);m.eval();return m,d


def pnp_reconstruct(m,R,y,steps=12,eta=.65,ridge=1e-3):
    B=y.shape[0];Rb=R.unsqueeze(0).expand(B,-1,-1);x=tikhonov(Rb,y,1e-3)
    for _ in range(steps):
        z=m(x);x=(z+eta*prox_disp(Rb,y,z,ridge)).clamp_min(0)
    return x


class SharedMGDUN(nn.Module):
    """Six-stage operator-aware vector-domain deep-unfolding control."""
    def __init__(self,width=48,stages=6):
        super().__init__();self.stages=stages
        self.prior=nn.Sequential(nn.Conv1d(3,width,5,padding=2),nn.GELU(),nn.Conv1d(width,width,5,padding=2),nn.GELU(),nn.Conv1d(width,1,5,padding=2))
        self.raw_step=nn.Parameter(torch.tensor(-2.0));self.raw_beta=nn.Parameter(torch.tensor(-2.3))
        k=torch.tensor([1.,4.,6.,4.,1.]);self.register_buffer('gk',(k/k.sum())[None,None])
    def forward(self,R,y):
        if R.ndim==2:R=R.unsqueeze(0).expand(y.shape[0],-1,-1)
        N=R.shape[-1];A=R/N;bp=torch.einsum('bmn,bm->bn',A,y);bp=bp/(bp.abs().amax(-1,keepdim=True)+1e-6)
        x=tikhonov(R,y,1e-3);base=F.softplus(self.raw_step);beta=torch.sigmoid(self.raw_beta)
        for k in range(self.stages):
            grad=torch.einsum('bmn,bm->bn',A,torch.einsum('bmn,bn->bm',A,x)-y)
            gx=torch.zeros_like(x);gx[:,1:-1]=.5*(x[:,2:]-x[:,:-2])
            prior=self.prior(torch.stack([x,bp,gx],1)).squeeze(1)
            step=base*(0.86**k)
            x=(x-step*grad+beta*prior).clamp_min(0)
            x=F.conv1d(x[:,None],self.gk,padding=2).squeeze(1).clamp_min(0)
        return x


def train_mgdun(seed,train_bank,val_items,out,device,quick=False):
    ck=Path(out)/f'MGDUN_seed{seed}.pt'
    m=SharedMGDUN().to(device)
    if ck.exists():
        d=torch.load(ck,map_location=device,weights_only=False);m.load_state_dict(d['model']);m.eval();return m,d
    steps=5 if quick else 950;torch.manual_seed(seed);random.seed(seed);wl=make_wavelength_grid(256,device=device);opt=torch.optim.AdamW(m.parameters(),lr=6e-4,weight_decay=1e-5)
    best=1e9;bad=0
    for step in range(1,steps+1):
        R=random.choice(train_bank);x=random_spectra(32 if not quick else 4,wl,family='random');nl=random.uniform(.004,.020);y=measure(R,x,nl);p=m(R,y)
        Rb=R.unsqueeze(0).expand(x.shape[0],-1,-1)
        loss=rel_l2(p,x).mean()+.015*sam(p,x).mean()+.01*forward_residual(Rb,p,y).mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
        if step%100==0 or step==steps:
            vals=[]
            with torch.no_grad():
                for d0 in val_items: vals.append(rel_l2(m(d0['R'],d0['val_y']),d0['val_x']))
                v=float(torch.cat(vals).mean())
            if v<best:best=v;bad=0;torch.save({'model':m.state_dict(),'best':best,'step':step,'seed':seed},ck)
            else:bad+=1
            if step>=650 and bad>=3:break
    d=torch.load(ck,map_location=device,weights_only=False);m.load_state_dict(d['model']);m.eval();return m,d


class ReSpecNN(nn.Module):
    """Public ReSpecNN topology adapted only to make the convolutional flatten dimension valid at N=256."""
    def __init__(self,M,N=256):
        super().__init__();self.M=M;self.N=N
        self.rec_fc=nn.Sequential(nn.Linear(M,1024),nn.ReLU(),nn.Dropout(.25),nn.Linear(1024,512),nn.ReLU(),nn.Dropout(.25),nn.Linear(512,N))
        self.conv=nn.Sequential(nn.Conv1d(1,8,25,padding=12),nn.MaxPool1d(3,2),nn.ReLU(),nn.Conv1d(8,32,11,padding=5),nn.MaxPool1d(3,2),nn.ReLU(),nn.Conv1d(32,64,5,padding=2),nn.MaxPool1d(3,2),nn.ReLU(),nn.Flatten())
        with torch.no_grad():flat=self.conv(torch.zeros(1,1,N)).shape[-1]
        self.fc=nn.Sequential(nn.Linear(flat,1024),nn.ReLU(),nn.Dropout(.25),nn.Linear(1024,512),nn.ReLU(),nn.Dropout(.25),nn.Linear(512,N))
    def forward(self,y):
        xt=self.rec_fc(y);return torch.sigmoid(self.fc(self.conv(xt[:,None]))+xt)


def train_respec(target, out, device, quick=False):
    name=target['name'];ck=Path(out)/f'RESPEC_{name}.pt';M,N=target['R'].shape;m=ReSpecNN(M,N).to(device)
    if ck.exists():
        d=torch.load(ck,map_location=device,weights_only=False);m.load_state_dict(d['model']);m.eval();return m,d
    steps=5 if quick else 1200;seed=26000+sum(ord(c) for c in name)%1000;torch.manual_seed(seed);random.seed(seed);wl=make_wavelength_grid(N,device=device);opt=torch.optim.AdamW(m.parameters(),lr=5e-4,weight_decay=1e-5);best=1e9;bad=0
    R=target['R']
    for step in range(1,steps+1):
        x=random_spectra(64 if not quick else 4,wl,family='random');nl=random.uniform(.004,.020)
        Rp=(R*(1+.03*torch.randn_like(R))).clamp_min(0);y=measure(Rp,x,nl);p=m(y);loss=rel_l2(p,x).mean()+.01*sam(p,x).mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
        if step%100==0 or step==steps:
            with torch.no_grad():v=float(rel_l2(m(target['val_y']),target['val_x']).mean())
            if v<best:best=v;bad=0;torch.save({'model':m.state_dict(),'best':best,'step':step,'M':M,'N':N,'seed':seed},ck)
            else:bad+=1
            if step>=800 and bad>=3:break
    d=torch.load(ck,map_location=device,weights_only=False);m.load_state_dict(d['model']);m.eval();return m,d
