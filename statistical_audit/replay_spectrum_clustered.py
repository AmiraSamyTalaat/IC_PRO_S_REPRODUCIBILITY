#!/usr/bin/env python3
"""Independent A12 spectrum-clustered reanalysis of archived case-level errors.

Each synthetic test spectrum occurs under TWO operators within its regime.
Group by (regime, sample_id), take operator-mean per method for each spectrum,
then compare paired spectrum-cluster differences. This corrects unit-of-sampling
for a finite tested operator pair; does NOT infer unseen-operator performance.
IC-PRO-S benchmark uses selected V7 seed 7; methods with multiple archived
seeds are averaged at (regime, operator, sample, method) as in A12 historical.
"""
from pathlib import Path
import argparse, collections, csv, json, numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

def read(file):
 with open(file,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--results-dir',type=Path, default=Path(__file__).resolve().parent/'a12_frozen_outputs')
 p.add_argument('--out',type=Path, default=Path(__file__).resolve().parent/'results')
 p.add_argument('--n-boot',type=int,default=20000)
 p.add_argument('--seed',type=int,default=20261015)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 rows=read(a.results_dir/'sample_errors_core.csv')+read(a.results_dir/'sample_errors_extra.csv')
 case=collections.defaultdict(list)
 for r in rows:
  if r['method']=='IC-PRO-S' and str(r['seed'])!='7': continue
  case[(r['regime'],r['operator'],int(r['sample']),r['method'])].append(float(r['rel_l2']))
 perop={k:float(np.mean(v)) for k,v in case.items()}
 clusters=collections.defaultdict(list)
 for (regime,operator,sample,method),v in perop.items():clusters[(regime,sample,method)].append((operator,v))
 collapsed={}
 for k,values in clusters.items():
  if len(values)!=2:raise RuntimeError(f'Expected 2 operators for spectrum-cluster {k}: {values}')
  collapsed[k]=float(np.mean([z for _,z in values]))
 regimes=sorted({k[0] for k in collapsed})
 expected_comparators=['NNLS-TV','PnP tuned','ReSpecNN per-device','PiGDM controlled','MGDUN-style']
 rg_byop=collections.defaultdict(set)
 for (rg,op,smp,method) in perop:rg_byop[rg].add(op)
 for rg in regimes:
  assert len(rg_byop[rg])==2
  for method in ['IC-PRO-S']+expected_comparators:
   assert len([s for r,s,m in collapsed if r==rg and m==method])==128,(rg,method)
 rng=np.random.default_rng(a.seed); outputs=[]
 for rg in ['interpolation','moderate_M_extrapolation','severe_structural']:
  samples=range(128); ic=np.array([collapsed[(rg,s,'IC-PRO-S')] for s in samples]);
  for comp in expected_comparators:
   cm=np.array([collapsed[(rg,s,comp)] for s in samples]);dif=cm-ic
   pval=float(wilcoxon(dif,alternative='greater').pvalue)
   boot_indices=rng.integers(0,len(dif),size=(a.n_boot,len(dif)))
   means=dif[boot_indices].mean(axis=1)
   lo,hi=np.quantile(means,[0.025,0.975]);
   outputs.append({'regime':rg,'comparator':comp,'n_distinct_spectra':128,'n_operators':2,
   'n_measurements':256,'comparator_minus_icpros':float(np.mean(dif)),
   'cluster_bootstrap_ci_low':float(lo),'cluster_bootstrap_ci_high':float(hi),
   'spectrum_cluster_wilcoxon_one_sided_p':pval,
   'fraction_distinct_spectra_comparator_worse':float(np.mean(dif>0))})
 adjusted=multipletests([x['spectrum_cluster_wilcoxon_one_sided_p'] for x in outputs],method='holm')[1]
 for x,adj in zip(outputs,adjusted):x['holm15_adjusted_p']=float(adj)
 with (a.out/'A12_CLUSTERED_15_COMPARISONS.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=outputs[0].keys());w.writeheader();w.writerows(outputs)
 methodmeans=[]
 for rg in regimes:
  methods=sorted({m for (r,_,m) in collapsed if r==rg})
  for method in methods:
   v=np.array([collapsed[(rg,s,method)] for s in range(128)])
   methodmeans.append({'regime':rg,'method':method,'n_spectra':128,'n_measurements':256,'mean_rel_l2':float(v.mean())})
 with (a.out/'A12_CLUSTERED_MEANS.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=methodmeans[0].keys());w.writeheader();w.writerows(methodmeans)
 sm={'status':'PASS','source':'Original A12 archived per-case errors; no model retraining',
 'unit':'128 unique spectra per regime, each evaluated by 2 frozen operators',
 'n_test_evaluations_per_regime':256,'n_spectrum_clusters_per_regime':128,
 'regimes':regimes,'tests':len(outputs),'bootstrap':a.n_boot,'bootstrap_rng_seed':a.seed,
 'multiplicity':'Holm across 15 tests','inference_scope':'Fixed operator pair only; NOT instrument-population sampling'}
 (a.out/'SUMMARY.json').write_text(json.dumps(sm,indent=2))
 print(json.dumps(sm,indent=2))
 for o in outputs:print(o['regime'][:13],o['comparator'][:14],f"diff={o['comparator_minus_icpros']:.5f}",f"CI={o['cluster_bootstrap_ci_low']:.5f},{o['cluster_bootstrap_ci_high']:.5f}",f"p={o['spectrum_cluster_wilcoxon_one_sided_p']:.4g}",f"Holm={o['holm15_adjusted_p']:.4g}")
if __name__=='__main__':main()
