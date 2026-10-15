#!/usr/bin/env python3
"""GPU-ready matched-parameter IC-PRO-S feature ablation.

IMPORTANT: Newly proposed protocol, NOT completed experimental evidence. The
same FiRNO architecture/parameter count is used in all arms. The removed
response-derived channels are replaced with fixed spectral-coordinate channels
BEFORE the first convolution, while the analytic anchor, singular-mode gate and
proximal stage remain unchanged. Every first-layer channel remains active. This only
isolates explicit neural input conditioning; it does NOT remove all A usage.
Historical V7 original raw tensors are unavailable. The independent validation
and test tensors are from the preserved SOURCE-DEFINED REPLAY SET.
"""
import argparse, csv, copy, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'work/01_MAIN_V7/N05V7SRC'))
from src.core import FiRNO,rel_l2,sam
from src.synthetic_v2 import random_spectra,measure,batch_for_instrument,make_wavelength_grid

ARMS=('full', 'no_fisher', 'anchor_grad_only')

def attach_mask(model,condition):
    def control_features(module, inputs):
        if condition=='full':return inputs
        a=inputs[0].clone();n=a.shape[-1]
        pos=torch.linspace(-1.,1.,n,device=a.device,dtype=a.dtype).view(1,1,n)
        if condition=='anchor_grad_only': a[:,1:2,:]=pos.expand(a.shape[0],1,n)
        a[:,3:4,:]=torch.cos(math.pi*(pos+1)).expand(a.shape[0],1,n)
        return (a,)
    return model.inp.register_forward_pre_hook(control_features)

def evaluate(model, dataset,device):
    model.eval();scores=[]
    with torch.no_grad():
        for R,y,x in dataset:
            for k in range(0,len(y),32):
                pred=model(R[k:k+32].to(device),y[k:k+32].to(device),0.01)['x']
                scores.extend(rel_l2(pred,x[k:k+32].to(device)).cpu().tolist())
    return float(np.mean(scores))

def train_one(condition, seed, tdata, valdata, testdata,steps,device,outdir):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model=FiRNO(width=64,mode='fisher_gate',prox=True,use_fisher_channel=True).to(device)
    mask=attach_mask(model,condition)
    params=sum(p.numel() for p in model.parameters());assert params==291620,(params,'Unexpected full FiRNO architecture')
    opt=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=1e-5)
    best=1e10;best_step=0;best_state=None;bad=0
    t0=time.monotonic(); wl=make_wavelength_grid(256,device=device)
    for step in range(1,steps+1):
        model.train()
        # identical batch schedule across conditions for fixed seed
        torch.manual_seed(20_000_000 + seed*10000+step)
        random.seed(20_000_000 + seed*10000+step)
        index=random.randrange(len(tdata))
        R=tdata[index].to(device)
        x=random_spectra(32,wl,family='random')
        noise=random.uniform(.004,.020)
        y=measure(R,x,noise); Rb=batch_for_instrument(R,x)
        out=model(Rb,y,noise)
        loss=rel_l2(out['x'],x).mean()+0.015*sam(out['x'],x).mean()+0.001*out['raw'].square().mean()
        opt.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if step%100==0 or step==steps:
            validation=evaluate(model,valdata,device)
            if validation<best:
                best=validation;best_step=step;best_state={k:v.cpu().clone() for k,v in model.state_dict().items()};bad=0
            else:bad+=1
            print(f'{condition} seed={seed} step={step} val={validation:.5f}',flush=True)
            if step>=650 and bad>=3:break
    if best_state is not None: model.load_state_dict(best_state)
    torch.save({'model':model.state_dict(),'arm':condition,'seed':seed,'steps':step,
                'best_step':best_step,'best_validation_rel_l2':best,
                'params':params,'protocol':'V7_SOURCE_DEFINED_REPLAY_NEW_CONTROL'},
               Path(outdir)/f'CAPACITY_{condition}_seed{seed}.pt')
    score=evaluate(model,testdata,device)
    mask.remove()
    return {'arm':condition,'seed':seed,'params':params,'steps':step,'best_step':best_step,
            'validation_rel_l2':best,'test_rel_l2':score,'seconds':time.monotonic()-t0}

def main():
    p=argparse.ArgumentParser();p.add_argument('--replay',default='/mnt/data/IC_PRO_S_REVIEWER_WORK_15OCT2026/V7_SOURCE_DEFINED_MAIN_EVAL_TENSORS.pt');p.add_argument('--out',default='/mnt/data/ICPRO_OCT15_NEXT/results');p.add_argument('--steps',type=int,default=1050);p.add_argument('--seeds',type=int,nargs='+',default=[3,7,11]);p.add_argument('--arms',choices=list(ARMS),nargs='+',default=list(ARMS));p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');args=p.parse_args()
    device='cuda' if (args.device=='auto' and torch.cuda.is_available()) else ('cpu' if args.device=='auto' else args.device)
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(2)
    d=torch.load(args.replay,map_location='cpu',weights_only=True)
    assert d['historical_raw_tensor_identity_claim'] is False
    tdata=d['bank']['train']; val=d['val'];test=d['test']
    rows=[];Path(args.out).mkdir(parents=True,exist_ok=True)
    for seed in args.seeds:
        for arm in args.arms:
            rows.append(train_one(arm,seed,tdata,val,test,args.steps,device,args.out))
            pd=Path(args.out)/'NEW_CAPACITY_ABLATION_RESULTS.csv'
            with open(pd,'w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(json.dumps({'result_rows':len(rows),'device':device,'status':'COMPLETE_FULL' if args.steps>=650 else 'SMOKE_TEST_ONLY'},indent=2))
if __name__=='__main__':main()
