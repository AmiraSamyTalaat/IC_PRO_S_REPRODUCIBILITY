import sys,json,math,os
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
sys.path.insert(0,'/mnt/data/N05_STRONG_Q1_EXECUTION_V2/code')
sys.path.insert(0,'/mnt/data/work_v7_source/project')
import torch
from strong_q1_models import ICPROFastFromFiRNO
from src.core import FiRNO,tikhonov

R=np.loadtxt('/mnt/data/response_matrix.txt').astype(np.float32);scale=float(np.max(np.abs(R)));A=R/scale;M,N=A.shape;grid=np.arange(N);rng=np.random.default_rng(20260812)

def spectra(B,mode='nominal'):
 X=[]
 for _ in range(B):
  if mode=='nominal':
   k=int(rng.integers(1,4));cs=[]
   while len(cs)<k:
    c=int(rng.integers(0,205))
    if all(abs(c-q)>=20 for q in cs):cs.append(c)
   x=np.zeros(N)
   for c in cs:
    w=rng.uniform(10,25);mag=rng.uniform(.6,1.4);q=w/(np.pi*((grid-c)**2+w*w));q/=q.max();x+=mag*q
  else:
   # structurally shifted comb/step spectra
   x=np.zeros(N);k=int(rng.integers(4,10))
   for c in rng.choice(np.arange(5,N-5),size=k,replace=False):
    w=rng.uniform(1.0,4.0);x+=rng.uniform(.3,1.2)*np.exp(-.5*((grid-c)/w)**2)
   if rng.random()<.5:x+=.3*(grid>rng.integers(40,220))
  x-=x.min();x/=x.max()+1e-12;X.append(x)
 return np.array(X,np.float32)

def meas(X,kind='id'):
 if kind=='id':
  Rn=A[None,:,:]*(1+.05*rng.standard_normal((len(X),M,N)).astype(np.float32));Y=np.einsum('bmn,bn->bm',Rn,X)+1e-5*np.maximum(rng.standard_normal((len(X),M)),0)
 elif kind=='response':
  Rn=A[None,:,:]*(1+.25*rng.standard_normal((len(X),M,N)).astype(np.float32));Y=np.einsum('bmn,bn->bm',Rn,X)
 elif kind=='dropout':
  Y=X@A.T
  for b in range(len(X)):Y[b,rng.choice(M,size=M//2,replace=False)]=0
 elif kind=='prior':Y=X@A.T
 elif kind=='noise':Y=X@A.T+.002*rng.standard_normal((len(X),M))+.001*rng.random((len(X),1))
 return np.maximum(Y,1e-12)

def normlog(Y):
 Z=np.log(np.clip(Y,1e-15,None));return (Z-Z.min(1,keepdims=True))/(Z.max(1,keepdims=True)-Z.min(1,keepdims=True)+1e-12)
# Train score models on simulation only
Xtr=spectra(5000);Ytr=meas(Xtr,'id');Ftr=normlog(Ytr)
Xid=spectra(800);Yid=meas(Xid,'id');Fid=normlog(Yid)
parts=[]
for kind in ['response','dropout','prior','noise']:
 X=spectra(200,'shift' if kind=='prior' else 'nominal');Y=meas(X,kind);parts.append((kind,Y,normlog(Y)))
Yood=np.concatenate([p[1] for p in parts]);Food=np.concatenate([p[2] for p in parts]);labels=np.r_[np.zeros(len(Fid),int),np.ones(len(Food),int)]
# scores
knn1=NearestNeighbors(n_neighbors=1).fit(Ftr);knn10=NearestNeighbors(n_neighbors=10).fit(Ftr)
mu=Ftr.mean(0);C=np.cov(Ftr,rowvar=False)+1e-4*np.eye(M);Ci=np.linalg.inv(C)
iso=IsolationForest(n_estimators=250,contamination='auto',random_state=7,n_jobs=-1).fit(Ftr)
def score_all(F):
 d1=knn1.kneighbors(F,return_distance=True)[0][:,0];d10=knn10.kneighbors(F,return_distance=True)[0].mean(1);dm=np.einsum('bi,ij,bj->b',F-mu,Ci,F-mu);si=-iso.score_samples(F);return {'nearest_manifold':d1,'knn_density':d10,'mahalanobis':dm,'isolation_forest':si}
SID=score_all(Fid);SOOD=score_all(Food)
# forward residual score via response-adapted IC-PRO checkpoint
ck=torch.load('/mnt/data/work_practical/checkpoints/FIRNO_EXTERNAL_RESPONSE_ONLY_ADAPTED.pt',map_location='cpu',weights_only=False);orig=FiRNO(**ck['cfg']);orig.load_state_dict(ck['model']);orig.eval();model=ICPROFastFromFiRNO(orig).eval();torch.set_num_threads(5)
Rm=torch.tensor(N*A,dtype=torch.float32)
@torch.no_grad()
def fwscore(Y,batch=64):
 out=[]
 for st in range(0,len(Y),batch):
  yy=torch.tensor(Y[st:st+batch],dtype=torch.float32);Rb=Rm.unsqueeze(0).expand(len(yy),-1,-1).contiguous();xr=tikhonov(Rb,yy,1e-3);amp=xr.amax(-1).clamp_min(1e-12);yc=yy/amp[:,None];p=model(Rb,yc,.01)['x'];Ap=torch.einsum('bmn,bn->bm',Rb/N,p);r=torch.linalg.vector_norm(Ap-yc,dim=-1)/(torch.linalg.vector_norm(yc,dim=-1)+1e-8);out.append(r.numpy())
 return np.concatenate(out)
SID['forward_residual']=fwscore(Yid);SOOD['forward_residual']=fwscore(Yood)

def fpr95(y,s):
 fpr,tpr,thr=roc_curve(y,s);idx=np.where(tpr>=.95)[0];return float(fpr[idx[0]]) if len(idx) else 1.0
metrics=[]
for k in SID:
 s=np.r_[SID[k],SOOD[k]];metrics.append({'score':k,'AUROC':float(roc_auc_score(labels,s)),'AUPRC':float(average_precision_score(labels,s)),'FPR95':fpr95(labels,s)})
# per OOD type AUROC
pertype={}
off=0
for kind,Y,F in parts:
 n=len(F);pertype[kind]={}
 for k in SID:
  s=np.r_[SID[k],SOOD[k][off:off+n]];y=np.r_[np.zeros(len(Fid)),np.ones(n)];pertype[kind][k]=float(roc_auc_score(y,s))
 off+=n
# real 9-single risk-coverage; real labels used only for post-hoc evaluation, never score calibration
sel=json.loads(Path('/mnt/data/work_practical/reports/SELECTIVE_SIM2REAL_GUARD.json').read_text());single={r['sample']:r for r in sel['single_rows']}
real_keys=[r['sample'] for r in sel['single_rows']];Yreal=np.array([np.loadtxt(f'/mnt/data/real_public/Ip/Ip_{k}.txt')/scale for k in real_keys]);Freal=normlog(Yreal);Sreal=score_all(Freal);Sreal['forward_residual']=fwscore(Yreal)
# errors for learned and fallback exactly from stored selected rows
learn_err=np.array([abs(r['learned_peak_nm']-r['ground_truth_nm'])/r['ground_truth_nm']*100 for r in sel['single_rows']])
fall_err=np.array([abs(r['nnls_tv_peak_nm']-r['ground_truth_nm'])/r['ground_truth_nm']*100 for r in sel['single_rows']])
risk_curves={}
for k,s in Sreal.items():
 order=np.argsort(-s) # reject highest score first
 curve=[]
 for rej in range(0,len(s)+1):
  rejected=set(order[:rej]);errs=np.array([fall_err[i] if i in rejected else learn_err[i] for i in range(len(s))]);coverage=1-rej/len(s);curve.append({'coverage_learned':coverage,'selective_MAE_pct':float(errs.mean()),'accepted_learned_MAE_pct':float(np.mean([learn_err[i] for i in range(len(s)) if i not in rejected])) if rej<len(s) else None})
 risk_curves[k]=curve
report={'simulation_only_ood_benchmark':{'id_n':len(Fid),'ood_n':len(Food),'ood_types':[p[0] for p in parts],'metrics':metrics,'per_type_AUROC':pertype},'real_single_peak_risk_coverage':risk_curves,'real_score_values':{k:{real_keys[i]:float(v[i]) for i in range(len(v))} for k,v in Sreal.items()},'governance':'All OOD detector fitting/calibration uses simulation only. Real labels are used only after scoring to report risk-coverage.'}
Path('/mnt/data/N05_STRONG_Q1_EXECUTION_V2/reports/OOD_DETECTOR_BENCHMARK.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['simulation_only_ood_benchmark'],indent=2))
