import sys,json,math,os
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest






R=np.loadtxt(str(Path(__file__).resolve().parents[1] / 'source' / 'HISTORICAL_response_matrix.txt')).astype(np.float32);scale=float(np.max(np.abs(R)));A=R/scale;M,N=A.shape;grid=np.arange(N);rng=np.random.default_rng(20260812)

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

# The original script's forward_residual requires a checkpoint and working directory
# not contained in the newly recovered zip. This replay intentionally covers
# the four pure-simulation detector scores ONLY, not the routing guard.
rows=[]
labels=np.r_[np.zeros(len(Fid),int),np.ones(len(Food),int)]
for key in SID:
    vals=np.r_[SID[key],SOOD[key]]
    fpr,tpr,_=roc_curve(labels,vals)
    ind=np.where(tpr>=.95)[0]
    rows.append({'score':key,'AUROC':float(roc_auc_score(labels,vals)),
                 'AUPRC':float(average_precision_score(labels,vals)),
                 'FPR95':float(fpr[ind[0]]) if len(ind) else 1.0})
from pathlib import Path
out=Path(__file__).resolve().parents[1]/'analysis'/'INDEPENDENT_DETECTOR_REPLAY_4_SCORES.json'
out.write_text(json.dumps({'n_id':len(Fid),'n_ood':len(Food),'reference_bank_n':len(Ftr),'metrics':rows,'note':'Simulated ID/OOD discrimination only; no 12000/600 guard threshold or selective routing'},indent=2))
print(json.dumps(rows,indent=2))
