from __future__ import annotations
import argparse, json, math, random, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

def parse():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--quick',action='store_true')
    return ap.parse_args()

A=parse()
BASE=Path(A.base).resolve()
OUT=Path(A.out).resolve()
OUT.mkdir(parents=True,exist_ok=True)
QUICK=bool(A.quick)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not QUICK and DEVICE.type!='cuda':
    raise RuntimeError('A14 FULL run requires CUDA.')

import sys
sys.path.insert(0,str(BASE))
sys.path.insert(0,str(BASE/'scripts'))
from src.core import FiRNO,tikhonov,rel_l2,forward_residual
from scripts.benchmark import build as build_a12

SEEDS=[3,7,11,19,23]
EPS_LEVELS=[0.0,0.01,0.025,0.05,0.10,0.20] if not QUICK else [0.0,0.05]
NOISE_STD=0.01
PERTURB_BASE_SEED=74001
METHODS=['IC-PRO-S learned eta','IC-PRO-S fixed eta=0.5','IC-PRO-S full eta=1',
         'IC-PRO-S proposal only','Tikhonov']

def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def load_ic(seed):
    p=BASE/'checkpoints'/f'FINAL_FIRNO_GATE_PROX_seed{seed}.pt'
    d=torch.load(p,map_location=DEVICE,weights_only=False)
    m=FiRNO(**d['cfg']).to(DEVICE);m.load_state_dict(d['model']);m.eval()
    return m,d

def smooth_calibration_perturb(R,eps,seed):
    if eps==0:
        return R.clone(),0.0
    st=torch.random.get_rng_state();torch.manual_seed(seed)
    g=torch.randn_like(R)
    k=torch.tensor([1,2,3,4,5,4,3,2,1],device=R.device,dtype=R.dtype)
    k=k/k.sum()
    g=F.conv1d(g[:,None,:],k[None,None,:],padding=4).squeeze(1)
    g=F.conv1d(g[:,None,:],k[None,None,:],padding=4).squeeze(1)
    g=(g-g.mean(-1,keepdim=True))/(g.std(-1,keepdim=True)+1e-6)
    Rt=(R*(1.0+eps*g)).clamp_min(1e-6)
    Rt=Rt/(Rt.amax(-1,keepdim=True)+1e-8)
    torch.random.set_rng_state(st)
    actual=float(torch.linalg.vector_norm(Rt-R)/torch.linalg.vector_norm(R).clamp_min(1e-12))
    return Rt,actual

def proposal_components(m,R,y,noise_std=NOISE_STD):
    # Exact replication of FiRNO.forward up to the proposal z, without changing weights.
    x0=tikhonov(R,y,1e-3)
    Af=R/R.shape[-1]
    bp=torch.einsum('bmn,bm->bn',Af,y)
    bp=bp/(bp.abs().amax(-1,keepdim=True)+1e-6)
    grad=torch.zeros_like(x0);grad[:,1:-1]=0.5*(x0[:,2:]-x0[:,:-2])
    feats=[x0,bp,grad]
    fmap=m.fisher_map(R,noise_std)
    if m.use_fisher_channel:feats.append(fmap)
    h=m.inp(torch.stack(feats,1))
    for b in m.blocks:h=b(h)
    raw=torch.tanh(m.out(h).squeeze(1))
    raw=(0.60*torch.sigmoid(m.step))*raw
    d,gates,S,dnull,drow=m.split_and_gate(R,y,x0,raw,noise_std)
    z=x0+d
    return z,S,gates

def exact_corr(z,R,y,noise_std=NOISE_STD):
    Af=R/R.shape[-1]
    resid=y-torch.einsum('bmn,bn->bm',Af,z)
    G=torch.einsum('bmn,bkn->bmk',Af,Af)
    I=torch.eye(G.shape[-1],device=G.device,dtype=G.dtype).expand_as(G)
    ridge=max(float(noise_std)**2,1e-6)
    alpha=torch.linalg.solve(G+ridge*I,resid.unsqueeze(-1)).squeeze(-1)
    return torch.einsum('bmn,bm->bn',Af,alpha)

def learned_eta_and_pre(m,z,R,y,S,gates,noise_std=NOISE_STD):
    pre,eta=m.proximal(z,R,y,S,gates,noise_std)
    return pre,eta

def rel_output_drift(p,p0):
    return torch.linalg.vector_norm(p-p0,dim=-1)/torch.linalg.vector_norm(p0,dim=-1).clamp_min(1e-8)

def theoretical_note():
    return r"""# A14 analytical stability note: exact quadratic proximal reconciliation

For a fixed calibrated operator \(A\), measurement \(y\), and proximal step \(\gamma>0\), define

\[
P_{A,y}(z)=\operatorname{prox}_{\gamma g_{A,y}}(z)
=(I+\gamma A^\top A)^{-1}(z+\gamma A^\top y),
\qquad
g_{A,y}(x)=\tfrac12\|Ax-y\|_2^2.
\]

The IC-PRO-S reconciliation with a fixed relaxation \(\eta\in[0,1]\) is

\[
T_{\eta,A,y}(z)=\Pi_+\!\left[(1-\eta)z+\eta P_{A,y}(z)\right].
\]

## 1. Nonexpansiveness in the learned proposal

The proximal map of a proper closed convex function is firmly nonexpansive, hence
\(\|P(z_1)-P(z_2)\|_2\le \|z_1-z_2\|_2\).
A convex combination of the identity with a nonexpansive map is nonexpansive for
\(\eta\in[0,1]\), and the nonnegative projection \(\Pi_+\) is also nonexpansive. Therefore

\[
\|T_{\eta,A,y}(z_1)-T_{\eta,A,y}(z_2)\|_2\le \|z_1-z_2\|_2.
\]

## 2. Measurement perturbation bound

Because \(H_A=I+\gamma A^\top A\succeq I\), \(\|H_A^{-1}\|_2\le1\). For a measurement perturbation
\(\delta y\),

\[
P_{A,y+\delta y}(z)-P_{A,y}(z)
=\gamma H_A^{-1}A^\top\delta y,
\]

so

\[
\|P_{A,y+\delta y}(z)-P_{A,y}(z)\|_2
\le \gamma\|A\|_2\|\delta y\|_2.
\]

For fixed \(\eta\), the corresponding under-relaxed and projected output perturbation is bounded by the same
quantity multiplied by \(\eta\).

## 3. Operator perturbation bound

Let \(B=A+E\), \(p_A=P_{A,y}(z)\), and \(H_B=I+\gamma B^\top B\). Then

\[
H_B(P_{B,y}(z)-p_A)
=\gamma\left[(A^\top A-B^\top B)p_A+(B^\top-A^\top)y\right].
\]

Using \(\|H_B^{-1}\|_2\le1\) and
\(\|A^\top A-B^\top B\|_2\le(\|A\|_2+\|B\|_2)\|E\|_2\),

\[
\|P_{B,y}(z)-P_{A,y}(z)\|_2
\le \gamma\|E\|_2
\left[(\|A\|_2+\|B\|_2)\|p_A\|_2+\|y\|_2\right].
\]

For a fixed \(\eta\), the under-relaxed reconciliation scales this envelope by \(\eta\).
If the learned reliability changes from \(\eta_A\) to \(\eta_B\), before projection,

\[
\|\tilde T_B-\tilde T_A\|_2
\le \eta_B\|P_B-P_A\|_2
+|\eta_B-\eta_A|\,\|P_A-z\|_2.
\]

Thus the learned gate has a clear interpretation: it controls the amount of operator/measurement sensitivity introduced
by the exact data-fidelity reconciliation while retaining an explicit path toward the proximal point.

A14 is the matched empirical perturbation test for this analytical envelope. It does not alter or retrain IC-PRO-S.
"""

def main():
    print('A14 DEVICE',DEVICE,'QUICK',QUICK,flush=True)
    # Re-use the already fixed A12 benchmark operators/spectra; no retraining.
    data=build_a12(DEVICE,quick=QUICK)
    (OUT/'THEORY_STABILITY_NOTE.md').write_text(theoretical_note(),encoding='utf-8')
    protocol={
      'status':'FROZEN_BEFORE_RESULTS',
      'benchmark':'A14 proximal stability under norm-controlled smooth multiplicative calibration perturbations',
      'base_benchmark':'A12_V1 operators and test spectra',
      'seeds':SEEDS,
      'epsilon_requested':EPS_LEVELS,
      'noise_std_in_fixed_A12_measurements':NOISE_STD,
      'measurement_policy':'test y remains generated from the unperturbed true operator; reconstruction receives perturbed calibrated operator',
      'perturbation':'two-pass smoothed multiplicative row perturbation followed by positivity clamp and row max normalization',
      'methods':METHODS,
      'training_or_finetuning':'none',
      'purpose':'mechanistic validation of exact proximal reconciliation and under-relaxation stability'
    }
    (OUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')

    models={s:load_ic(s)[0] for s in SEEDS}
    rows=[]
    ref_preds={} # (seed,operator,method)->cpu tensor at epsilon 0
    perturb_records=[]
    op_index=0
    for regime,items in data.items():
      for d in items:
        name=d['name'];Rtrue=d['R'];x=d['test_x'];y=d['test_y'];B=x.shape[0]
        op_index+=1
        for ei,eps in enumerate(EPS_LEVELS):
            Rt,actual=smooth_calibration_perturb(Rtrue,eps,PERTURB_BASE_SEED+op_index*100+ei)
            perturb_records.append({'regime':regime,'operator':name,'M':int(Rtrue.shape[0]),
                                    'epsilon_requested':eps,'relative_frobenius_actual':actual})
            # Tikhonov once per sample, independent of neural seed.
            for j in range(0,B,32 if not QUICK else 4):
                xb=x[j:j+32 if not QUICK else j+4];yb=y[j:j+32 if not QUICK else j+4]
                Rtb=Rt.unsqueeze(0).expand(xb.shape[0],-1,-1)
                Rtrueb=Rtrue.unsqueeze(0).expand(xb.shape[0],-1,-1)
                with torch.no_grad():
                    tp=tikhonov(Rtb,yb,1e-3)
                    er=rel_l2(tp,xb); fr_true=forward_residual(Rtrueb,tp,yb);fr_ass=forward_residual(Rtb,tp,yb)
                for k in range(xb.shape[0]):
                    rows.append({'regime':regime,'operator':name,'M':int(Rtrue.shape[0]),'epsilon_requested':eps,
                                 'relative_frobenius_actual':actual,'seed':'','sample':j+k,'method':'Tikhonov',
                                 'rel_l2':float(er[k]),'forward_residual_true_operator':float(fr_true[k]),
                                 'forward_residual_assumed_operator':float(fr_ass[k]),'eta':np.nan,'output_drift_from_eps0':np.nan})

            for seed,m in models.items():
                preds_for_eps={method:[] for method in METHODS if method!='Tikhonov'}
                etas=[]
                # batch inference
                for j in range(0,B,32 if not QUICK else 4):
                    stop=min(B,j+(32 if not QUICK else 4))
                    xb=x[j:stop];yb=y[j:stop]
                    Rtb=Rt.unsqueeze(0).expand(xb.shape[0],-1,-1)
                    Rtrueb=Rtrue.unsqueeze(0).expand(xb.shape[0],-1,-1)
                    with torch.no_grad():
                        z,S,gates=proposal_components(m,Rtb,yb)
                        corr=exact_corr(z,Rtb,yb)
                        learned_pre,eta=learned_eta_and_pre(m,z,Rtb,yb,S,gates)
                        preds={
                           'IC-PRO-S proposal only':z.clamp_min(0),
                           'IC-PRO-S fixed eta=0.5':(z+0.5*corr).clamp_min(0),
                           'IC-PRO-S full eta=1':(z+corr).clamp_min(0),
                           'IC-PRO-S learned eta':learned_pre.clamp_min(0)
                        }
                    for method,pred in preds.items():
                        preds_for_eps[method].append(pred.detach().cpu())
                        er=rel_l2(pred,xb);frt=forward_residual(Rtrueb,pred,yb);fra=forward_residual(Rtb,pred,yb)
                        for k in range(xb.shape[0]):
                            rows.append({'regime':regime,'operator':name,'M':int(Rtrue.shape[0]),'epsilon_requested':eps,
                                         'relative_frobenius_actual':actual,'seed':seed,'sample':j+k,'method':method,
                                         'rel_l2':float(er[k]),'forward_residual_true_operator':float(frt[k]),
                                         'forward_residual_assumed_operator':float(fra[k]),
                                         'eta':float(eta[k]) if method=='IC-PRO-S learned eta' else (0.0 if method=='IC-PRO-S proposal only' else (0.5 if method=='IC-PRO-S fixed eta=0.5' else 1.0)),
                                         'output_drift_from_eps0':np.nan})
                # compute output drift after concatenating prediction tensors
                for method,parts in preds_for_eps.items():
                    pcat=torch.cat(parts,0)
                    key=(seed,name,method)
                    if eps==0:
                        ref_preds[key]=pcat
                        drift=np.zeros(B,dtype=np.float32)
                    else:
                        p0=ref_preds[key]
                        drift=(torch.linalg.vector_norm(pcat-p0,dim=-1)/torch.linalg.vector_norm(p0,dim=-1).clamp_min(1e-8)).numpy()
                    # patch rows for this seed/operator/eps/method
                    idxs=[ii for ii,r in enumerate(rows) if r['operator']==name and r['epsilon_requested']==eps and r['seed']==seed and r['method']==method]
                    if len(idxs)!=B:
                        raise RuntimeError(f'row patch mismatch {name} {eps} {seed} {method}: {len(idxs)} != {B}')
                    for ii,dv in zip(idxs,drift):
                        rows[ii]['output_drift_from_eps0']=float(dv)

    df=pd.DataFrame(rows)
    df.to_csv(OUT/'PER_CASE.csv',index=False)
    pd.DataFrame(perturb_records).drop_duplicates().to_csv(OUT/'PERTURBATION_REALIZATION.csv',index=False)

    # Summary: neural modes averaged across matched seeds and samples; Tikhonov across samples.
    summary=[]
    for (reg,op,M,eps,method),g in df.groupby(['regime','operator','M','epsilon_requested','method']):
        summary.append({
          'regime':reg,'operator':op,'M':M,'epsilon_requested':eps,'method':method,'n_rows':len(g),
          'mean_rel_l2':g.rel_l2.mean(),'std_rel_l2':g.rel_l2.std(ddof=1),
          'mean_forward_residual_true_operator':g.forward_residual_true_operator.mean(),
          'mean_forward_residual_assumed_operator':g.forward_residual_assumed_operator.mean(),
          'mean_eta':g.eta.mean() if g.eta.notna().any() else np.nan,
          'mean_output_drift_from_eps0':g.output_drift_from_eps0.mean() if g.output_drift_from_eps0.notna().any() else np.nan,
          'relative_frobenius_actual':g.relative_frobenius_actual.iloc[0]
        })
    sdf=pd.DataFrame(summary)
    sdf.to_csv(OUT/'SUMMARY_BY_OPERATOR.csv',index=False)
    pooled=df.groupby(['epsilon_requested','method']).agg(
      n_rows=('rel_l2','count'),mean_rel_l2=('rel_l2','mean'),std_rel_l2=('rel_l2','std'),
      mean_forward_residual_true_operator=('forward_residual_true_operator','mean'),
      mean_forward_residual_assumed_operator=('forward_residual_assumed_operator','mean'),
      mean_eta=('eta','mean'),mean_output_drift_from_eps0=('output_drift_from_eps0','mean'),
      mean_actual_perturbation=('relative_frobenius_actual','mean')).reset_index()
    pooled.to_csv(OUT/'SUMMARY_POOLED.csv',index=False)

    # Matched learned-eta comparisons against proposal/fixed/full.
    from scipy.stats import wilcoxon
    stats=[]
    learned=df[df.method=='IC-PRO-S learned eta']
    for eps in EPS_LEVELS:
      L=learned[learned.epsilon_requested==eps][['regime','operator','seed','sample','rel_l2']].rename(columns={'rel_l2':'learned'})
      for comp in ['IC-PRO-S proposal only','IC-PRO-S fixed eta=0.5','IC-PRO-S full eta=1']:
        C=df[(df.epsilon_requested==eps)&(df.method==comp)][['regime','operator','seed','sample','rel_l2']].rename(columns={'rel_l2':'comp'})
        z=L.merge(C,on=['regime','operator','seed','sample'])
        diff=z.comp-z.learned
        rng=np.random.default_rng(1414+int(eps*10000));boots=[]
        for _ in range(3000 if not QUICK else 40):
            ix=rng.integers(0,len(diff),len(diff));boots.append(float(diff.iloc[ix].mean()))
        lo,hi=np.quantile(boots,[.025,.975])
        try:p=float(wilcoxon(diff,alternative='greater').pvalue)
        except Exception:p=float('nan')
        stats.append({'epsilon_requested':eps,'comparator':comp,'comparator_minus_learned_mean':diff.mean(),
                      'paired95_low':lo,'paired95_high':hi,'wilcoxon_one_sided_p_comparator_gt_learned':p})
    pd.DataFrame(stats).to_csv(OUT/'PAIRED_PROX_STATS.csv',index=False)

    # Empirical low-perturbation sensitivity slopes (epsilon <= 0.10), pooled.
    slopes=[]
    psub=pooled[pooled.epsilon_requested<=0.10]
    for method,g in psub.groupby('method'):
        if len(g)>=3:
            xeps=g.mean_actual_perturbation.values
            yerr=g.mean_rel_l2.values
            yd=g.mean_output_drift_from_eps0.values
            slopes.append({'method':method,
                           'rel_l2_slope_per_unit_actual_perturbation':float(np.polyfit(xeps,yerr,1)[0]),
                           'output_drift_slope_per_unit_actual_perturbation':float(np.polyfit(xeps,yd,1)[0]) if np.isfinite(yd).sum()>=3 else np.nan})
    pd.DataFrame(slopes).to_csv(OUT/'EMPIRICAL_SENSITIVITY_SLOPES.csv',index=False)

    # Seed stability of learned eta path.
    seedtab=df[df.method=='IC-PRO-S learned eta'].groupby(['epsilon_requested','seed']).agg(
        mean_rel_l2=('rel_l2','mean'),mean_eta=('eta','mean'),
        mean_output_drift=('output_drift_from_eps0','mean')).reset_index()
    seedtab.to_csv(OUT/'LEARNED_ETA_5SEED_STABILITY.csv',index=False)

    # Simple plots.
    import matplotlib.pyplot as plt
    for metric,ylabel,fname in [
        ('mean_rel_l2','Mean Relative-L2','FIG_ERROR_VS_PERTURBATION.png'),
        ('mean_output_drift_from_eps0','Output drift from eps=0','FIG_OUTPUT_DRIFT_VS_PERTURBATION.png'),
        ('mean_forward_residual_true_operator','True-operator forward residual','FIG_TRUE_FORWARD_RESIDUAL.png')]:
        fig=plt.figure(figsize=(8.8,5.2));ax=fig.add_subplot(111)
        for method,g in pooled.groupby('method'):
            ax.plot(g.mean_actual_perturbation,g[metric],marker='o',label=method)
        ax.set_xlabel('Actual relative Frobenius operator perturbation')
        ax.set_ylabel(ylabel);ax.legend(fontsize=8);ax.grid(alpha=.25);fig.tight_layout()
        fig.savefig(OUT/fname,dpi=220);plt.close(fig)
    le=pooled[pooled.method=='IC-PRO-S learned eta']
    fig=plt.figure(figsize=(7.2,4.6));ax=fig.add_subplot(111)
    ax.plot(le.mean_actual_perturbation,le.mean_eta,marker='o')
    ax.set_xlabel('Actual relative Frobenius operator perturbation');ax.set_ylabel('Mean learned eta')
    ax.set_title('Adaptive reconciliation strength under calibration perturbation');ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(OUT/'FIG_ETA_VS_PERTURBATION.png',dpi=220);plt.close(fig)

    required=set(METHODS)
    seen=set(df.method.unique())
    final={
      'status':'PASS' if (not QUICK and required.issubset(seen) and set(SEEDS).issubset(set(pd.to_numeric(df.seed,errors="coerce").dropna().astype(int).unique()))) else ('SMOKE_PASS' if QUICK else 'FAIL'),
      'quick_mode':QUICK,'device':str(DEVICE),'seeds':SEEDS,'epsilon_levels':EPS_LEVELS,
      'operators':int(df[['regime','operator']].drop_duplicates().shape[0]),
      'methods':sorted(seen),'training_or_finetuning':False,
      'base_benchmark':'A12_V1'
    }
    (OUT/'FINAL_STATUS.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
    print(json.dumps(final,indent=2),flush=True)
    if not QUICK and final['status']!='PASS':raise RuntimeError('A14 final gate failed')
    print('A14_COMPLETE_OK',flush=True)

if __name__=='__main__':
    main()
