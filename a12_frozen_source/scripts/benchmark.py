from __future__ import annotations
import math, json, hashlib
from pathlib import Path
import numpy as np
import torch
from src.synthetic_v2 import make_wavelength_grid, random_response, random_spectra, measure

N=256
NOISE=0.01
TEST_SPECTRA=128
VAL_SPECTRA=64


def _preserve_seed(seed):
    class Ctx:
        def __enter__(self):
            self.st=torch.random.get_rng_state(); torch.manual_seed(seed)
        def __exit__(self,*args): torch.random.set_rng_state(self.st)
    return Ctx()


def structural_response(m, wl, seed):
    """Structurally different positive filters: smoothed-random, triangular, high-frequency mixed."""
    with _preserve_seed(seed):
        N=wl.numel(); R=torch.zeros(m,N,device=wl.device)
        x=wl
        for i in range(m):
            typ=i%3
            if typ==0: # smoothed random
                q=torch.rand(N,device=wl.device)
                # repeated smooth convolution, much less Gaussian-mixture-like than training generator
                k=torch.tensor([1,2,3,4,5,4,3,2,1],dtype=q.dtype,device=q.device); k=k/k.sum()
                q=torch.nn.functional.conv1d(q[None,None],k[None,None],padding=4).squeeze()
                q=torch.nn.functional.conv1d(q[None,None],k[None,None],padding=4).squeeze()
                row=0.08+q
            elif typ==1: # triangular mixture
                row=torch.zeros(N,device=wl.device)
                for _ in range(3):
                    c=float(torch.empty(1).uniform_(0.05,0.95)); w=float(torch.empty(1).uniform_(0.035,0.16)); a=float(torch.empty(1).uniform_(0.4,1.0))
                    row += a*torch.clamp(1-torch.abs(x-c)/w,min=0)
                row += 0.04
            else: # high-frequency mixed
                row=torch.zeros(N,device=wl.device)
                for _ in range(3):
                    c=torch.rand(1,device=wl.device); s=torch.empty(1,device=wl.device).uniform_(0.008,0.045); a=torch.empty(1,device=wl.device).uniform_(0.3,1.0)
                    row += a*torch.exp(-0.5*((x-c)/s)**2)
                freq=torch.empty(1,device=wl.device).uniform_(18,34); phase=torch.rand(1,device=wl.device)*2*math.pi
                row *= (0.55+0.45*torch.cos(2*math.pi*freq*x+phase)).clamp_min(0.03)
                row += 0.03
            R[i]=row/row.max().clamp_min(1e-8)
        return R


def standard_response(m, wl, seed, complexity=5):
    with _preserve_seed(seed):
        return random_response(m,wl,complexity=complexity,family='train')


def operators(device):
    wl=make_wavelength_grid(N,device=device)
    return {
      'interpolation': [
        ('interp_M16_A', standard_response(16,wl,9101,5)),
        ('interp_M24_B', standard_response(24,wl,9102,6)),
      ],
      'moderate_M_extrapolation': [
        ('moderate_M8', standard_response(8,wl,9201,5)),
        ('moderate_M40', standard_response(40,wl,9202,6)),
      ],
      'severe_structural': [
        ('struct_M16', structural_response(16,wl,9301)),
        ('struct_M32', structural_response(32,wl,9302)),
      ]
    }


def _spectra(n,device,seed):
    with _preserve_seed(seed):
        wl=make_wavelength_grid(N,device=device)
        return random_spectra(n,wl,family='random')


def _meas(R,x,seed,noise=NOISE):
    with _preserve_seed(seed): return measure(R,x,noise)


def build(device, quick=False):
    ns_test=6 if quick else TEST_SPECTRA; ns_val=4 if quick else VAL_SPECTRA
    ops=operators(device); out={}
    for ri,(reg,items) in enumerate(ops.items()):
        xt=_spectra(ns_test,device,41001+ri); xv=_spectra(ns_val,device,40001+ri)
        out[reg]=[]
        for oi,(name,R) in enumerate(items):
            yt=_meas(R,xt,51001+ri*10+oi); yv=_meas(R,xv,50001+ri*10+oi)
            out[reg].append({'name':name,'R':R,'test_x':xt,'test_y':yt,'val_x':xv,'val_y':yv})
    return out


def save_manifest(data,path):
    p=Path(path); rows=[]
    h=hashlib.sha256()
    for reg,items in data.items():
        for d in items:
            for key in ['R','test_x','test_y','val_x','val_y']:
                a=d[key].detach().cpu().numpy().astype(np.float32)
                h.update(a.tobytes())
            rows.append({'regime':reg,'operator':d['name'],'M':int(d['R'].shape[0]),'N':int(d['R'].shape[1]),
                         'test_n':int(d['test_x'].shape[0]),'val_n':int(d['val_x'].shape[0])})
    obj={'version':'A12_V1','N':N,'noise':NOISE,'rows':rows,'content_sha256':h.hexdigest(),
         'test_spectrum_seeds':[41001,41002,41003], 'validation_spectrum_seeds':[40001,40002,40003],
         'operator_seeds':[9101,9102,9201,9202,9301,9302]}
    p.write_text(json.dumps(obj,indent=2),encoding='utf-8'); return obj
