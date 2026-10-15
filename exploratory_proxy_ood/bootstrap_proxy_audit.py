#!/usr/bin/env python3
"""Paired spectrum-cluster bootstrap for historical A12 support-score PROXY.
DO NOT interpret these statistics as results for the article's original OOD guard.
Uses existing archived proxy score and paired IC-PRO-S/NNLS-TV errors; no reruns, no labels used in routing.
"""
from pathlib import Path
import numpy as np, pandas as pd, argparse, json

def audit(df, nboot=20000, seed=20261015):
    df=df.copy()
    # One spectral input observed under two frozen operators per regime.
    # Cluster = (regime, sample). Resample within regimes, retaining both operators.
    regimes=sorted(df['regime'].unique())
    ops=sorted(df['operator'].unique())
    assert len(df)==768 and len(regimes)==3 and len(ops)==6
    assert not df[['regime','operator','sample']].duplicated().any()
    assert df['sample'].nunique()==128
    d=df['routed_error_99'].to_numpy()-df['IC-PRO-S'].to_numpy()
    assert ((df['accepted_99'].astype(bool)) | (abs(d)>1e-13)).all()
    arrays=[]; rejected=[]; raw=[]; routed=[]; is50=[]
    df['lowest_half_within_operator']=False
    for name,sub in df.groupby('operator',sort=False):
        ii=sub.nsmallest(64,'support_distance').index
        df.loc[ii,'lowest_half_within_operator']=True
    for reg in regimes:
        g=df[df.regime==reg].sort_values(['sample','operator'])
        assert len(g)==256
        for s in range(128):
            q=g[g['sample']==s]
            assert len(q)==2
        arrays.append(g['routed_error_99'].to_numpy().reshape(128,2)-g['IC-PRO-S'].to_numpy().reshape(128,2))
        rejected.append((~g['accepted_99'].astype(bool)).to_numpy().reshape(128,2))
        raw.append(g['IC-PRO-S'].to_numpy().reshape(128,2))
        routed.append(g['routed_error_99'].to_numpy().reshape(128,2))
        is50.append(g['lowest_half_within_operator'].to_numpy().reshape(128,2))
    arrays=np.stack(arrays); rejected=np.stack(rejected);raw=np.stack(raw);routed=np.stack(routed);is50=np.stack(is50)
    rng=np.random.default_rng(seed)
    results=[]
    for s in range(0,nboot,250):
        b=min(250,nboot-s)
        ix=rng.integers(0,128,size=(b,3,128))
        ix2=np.broadcast_to(ix[...,None],(b,3,128,2))
        dif=np.take_along_axis(arrays[None],ix2,axis=2)
        rej=np.take_along_axis(rejected[None],ix2,axis=2)
        sel=np.take_along_axis(is50[None],ix2,axis=2)
        vals=np.take_along_axis(raw[None],ix2,axis=2)
        rr=np.take_along_axis(routed[None],ix2,axis=2)
        improvement=-dif.mean(axis=(1,2,3))
        rejection=rej.mean(axis=(1,2,3))
        selected=(vals*sel).sum(axis=(1,2,3))/sel.sum(axis=(1,2,3))
        selected_delta=selected-vals.mean(axis=(1,2,3))
        results.append(np.stack([improvement,rejection,selected,selected_delta],axis=1))
    bs=np.concatenate(results,axis=0)
    out={
      'source_provenance':'Historical A12 inputs and case errors; exploratory support-score proxy, not original published guard',
      'independence_unit':'(regime,sample) = 384 unique spectra, 2 operators per spectrum, 128 independent spectra per regime',
      'number_of_operator_matrices':6,
      'n_case_evaluations':len(df),'n_independent_spectral_clusters':384,
      'selection':'99th-percentile separate simulation threshold; frozen case scores; offline NNLS-TV replacement for flagged cases',
      'n_flagged':int((~df.accepted_99.astype(bool)).sum()),
      'rejection_percent':float(100*(~df.accepted_99.astype(bool)).mean()),
      'mean_icpros':float(df['IC-PRO-S'].mean()),
      'mean_routed_counterfactual':float(df['routed_error_99'].mean()),
      'absolute_mean_gain_routing':float(-d.mean()),
      'routing_improvement_95CI_clustered':list(map(float,np.quantile(bs[:,0],[.025,.975]))),
      'flagged_fraction_95CI_clustered':list(map(float,np.quantile(bs[:,1],[.025,.975]))),
      'lowest_half_risk':float(df.loc[df.lowest_half_within_operator,'IC-PRO-S'].mean()),
      'lowest_half_vs_all_delta':float(df.loc[df.lowest_half_within_operator,'IC-PRO-S'].mean()-df['IC-PRO-S'].mean()),
      'lowest_half_delta_95CI_clustered':list(map(float,np.quantile(bs[:,3],[.025,.975]))),
      'nominal_bootstrap_resamples':nboot,'bootstrap_seed':seed,
      'essential_limitations':['not the published OOD guard','no true OOD labels','spectral clustering does not estimate population of operators','rerouting changes errors offline only, not measured end-to-end latency']}
    return out,df

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cases',required=True);ap.add_argument('--out',required=True);ap.add_argument('--nboot',type=int,default=20000)
    args=ap.parse_args();df=pd.read_csv(args.cases);out,df=audit(df,args.nboot)
    p=Path(args.out);p.mkdir(parents=True,exist_ok=True)
    (p/'CLUSTERED_PROXY_AUDIT.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
