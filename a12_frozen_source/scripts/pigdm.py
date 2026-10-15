from __future__ import annotations
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from src.synthetic_v2 import make_wavelength_grid, random_spectra


def timestep_embedding(t,dim):
    half=dim//2;freq=torch.exp(-math.log(10000)*torch.arange(half,device=t.device).float()/max(half-1,1));a=t.float()[:,None]*freq[None];e=torch.cat([torch.sin(a),torch.cos(a)],1)
    if dim%2:e=F.pad(e,(0,1))
    return e

class TRes(nn.Module):
    def __init__(self,ci,co,td,drop=.05):
        super().__init__();g1=max(1,min(8,ci));g2=max(1,min(8,co));
        while ci%g1:g1-=1
        while co%g2:g2-=1
        self.n1=nn.GroupNorm(g1,ci);self.c1=nn.Conv1d(ci,co,3,padding=1);self.te=nn.Linear(td,co);self.n2=nn.GroupNorm(g2,co);self.c2=nn.Conv1d(co,co,3,padding=1);self.drop=nn.Dropout(drop);self.skip=nn.Conv1d(ci,co,1) if ci!=co else nn.Identity()
    def forward(self,x,te):
        h=self.c1(F.silu(self.n1(x)))+self.te(te)[:,:,None];h=self.c2(self.drop(F.silu(self.n2(h))));return h+self.skip(x)

class UNet1D(nn.Module):
    """Four-level DDPM noise predictor, channel multipliers 1,2,4,8, two residual blocks/level + two bottleneck blocks."""
    def __init__(self,base=45,drop=.05):
        super().__init__();td=base*4;self.td=td;self.temb=nn.Sequential(nn.Linear(td,td),nn.SiLU(),nn.Linear(td,td));self.inp=nn.Conv1d(1,base,3,padding=1)
        ch=[base,base*2,base*4,base*8]
        self.dblocks=nn.ModuleList();self.down=nn.ModuleList();ci=base
        for i,co in enumerate(ch):
            self.dblocks.append(nn.ModuleList([TRes(ci,co,td,drop),TRes(co,co,td,drop)]));ci=co
            if i<3:self.down.append(nn.Conv1d(ci,ci,4,stride=2,padding=1))
        self.mid1=TRes(ci,ci,td,drop);self.mid2=TRes(ci,ci,td,drop)
        self.up=nn.ModuleList();self.ublocks=nn.ModuleList()
        for i in range(2,-1,-1):
            self.up.append(nn.ConvTranspose1d(ci,ch[i],4,stride=2,padding=1));ci=ch[i]
            self.ublocks.append(nn.ModuleList([TRes(ci+ch[i],ci,td,drop),TRes(ci,ci,td,drop)]))
        self.outn=nn.GroupNorm(4 if base%4==0 else 1,base);self.out=nn.Conv1d(base,1,3,padding=1)
    def forward(self,x,t):
        te=self.temb(timestep_embedding(t,self.td));h=self.inp(x[:,None]);sk=[]
        for i,b in enumerate(self.dblocks):
            h=b[0](h,te);h=b[1](h,te);sk.append(h)
            if i<3:h=self.down[i](h)
        h=self.mid2(self.mid1(h,te),te)
        for j in range(3):
            h=self.up[j](h);s=sk[2-j]
            if h.shape[-1]!=s.shape[-1]:h=F.interpolate(h,size=s.shape[-1],mode='linear',align_corners=False)
            h=torch.cat([h,s],1);h=self.ublocks[j][0](h,te);h=self.ublocks[j][1](h,te)
        return self.out(F.silu(self.outn(h))).squeeze(1)


def cosine_betas(T=1000,s=.008,device='cpu'):
    x=torch.linspace(0,T,T+1,device=device);ac=torch.cos(((x/T+s)/(1+s))*math.pi/2)**2;ac=ac/ac[0];b=1-ac[1:]/ac[:-1];return b.clamp(1e-5,.999)

class PiGDM:
    def __init__(self,device,T=1000,base=45):
        self.device=device;self.T=T;self.model=UNet1D(base=base).to(device);self.betas=cosine_betas(T,device=device);self.alphas=1-self.betas;self.ac=torch.cumprod(self.alphas,0)
    def train_or_load(self,out,quick=False):
        ck=Path(out)/'PIGDM_DDPM_step3000.pt'
        if ck.exists():
            d=torch.load(ck,map_location=self.device,weights_only=False);self.model.load_state_dict(d['model']);self.model.eval();return d
        steps=3 if quick else 3000;torch.manual_seed(2026);wl=make_wavelength_grid(256,device=self.device);opt=torch.optim.AdamW(self.model.parameters(),lr=2e-4,weight_decay=1e-5)
        # fixed validation bank
        st=torch.random.get_rng_state();torch.manual_seed(2027);vx=random_spectra(12 if quick else 1024,wl,family='random')*2-1;vt=torch.randint(0,self.T,(vx.shape[0],),device=self.device);ve=torch.randn_like(vx);torch.random.set_rng_state(st)
        best=1e9;beststep=0
        for step in range(1,steps+1):
            if step==2001 and not quick:
                for g in opt.param_groups:g['lr']=1e-4
            x=random_spectra(4 if quick else 32,wl,family='random')*2-1;t=torch.randint(0,self.T,(x.shape[0],),device=self.device);e=torch.randn_like(x);ab=self.ac[t][:,None];xt=torch.sqrt(ab)*x+torch.sqrt(1-ab)*e;pred=self.model(xt,t);loss=F.mse_loss(pred,e)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),1);opt.step()
            if step%250==0 or step==steps:
                self.model.eval();
                with torch.no_grad():ab=self.ac[vt][:,None];q=torch.sqrt(ab)*vx+torch.sqrt(1-ab)*ve;v=float(F.mse_loss(self.model(q,vt),ve))
                self.model.train()
                if v<best:best=v;beststep=step;torch.save({'model':self.model.state_dict(),'best_val_noise_mse':best,'step':step,'T':self.T,'base':self.model.inp.out_channels},ck)
        d=torch.load(ck,map_location=self.device,weights_only=False);self.model.load_state_dict(d['model']);self.model.eval();return d
    def _project_q0(self,q,R,y,gamma=.05,ridge=1e-6):
        # y=(R/N)x, x=(q+1)/2 => y=Hq+b, H=R/(2N), b=(R1)/(2N)
        N=R.shape[-1];H=R/(2*N);b=H@torch.ones(N,device=q.device);target=y-b[None]
        G=H@H.T+ridge*torch.eye(H.shape[0],device=q.device);res=target-q@H.T;alpha=torch.linalg.solve(G,res.T).T;corr=alpha@H
        return q+gamma*corr
    @torch.no_grad()
    def reconstruct(self,R,y,gamma=.05,K=50):
        B=y.shape[0];q=torch.randn(B,256,device=self.device);ts=torch.linspace(self.T-1,0,K,device=self.device).round().long().unique_consecutive()
        for i,tv in enumerate(ts):
            t=torch.full((B,),int(tv),device=self.device,dtype=torch.long);ab=self.ac[t][:,None];eps=self.model(q,t);q0=(q-torch.sqrt(1-ab)*eps)/torch.sqrt(ab);q0=q0.clamp(-1.5,1.5);q0=self._project_q0(q0,R,y,gamma)
            if i==len(ts)-1:q=q0;break
            tn=ts[i+1];abn=self.ac[tn];q=torch.sqrt(abn)*q0+torch.sqrt(1-abn)*eps
        return ((q.clamp(-1,1)+1)/2).clamp(0,1)
