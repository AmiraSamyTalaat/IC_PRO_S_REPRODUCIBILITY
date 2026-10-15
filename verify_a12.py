#!/usr/bin/env python3
from pathlib import Path
import csv, collections, math, json
root=Path(__file__).resolve().parent
out=root/'a12_frozen_outputs'
rows=[]
for basename in ('sample_errors_core.csv','sample_errors_extra.csv'):
 with (out/basename).open(newline='') as f:rows.extend(list(csv.DictReader(f)))
case=collections.defaultdict(list)
for r in rows:
 method=r['method']; seed=r['seed']
 if method=='IC-PRO-S' and seed!='7':continue
 case[(r['regime'],r['operator'],int(r['sample']),method)].append(float(r['rel_l2']))
value={key:sum(v)/len(v) for key,v in case.items()}
groups=collections.defaultdict(list)
for (regime,operator,sample,method),v in value.items():groups[(regime,method)].append(v)
with (out/'EXTENDED_TABLE6.csv').open(newline='') as f:expect=list(csv.DictReader(f))
counts=[]
for r in expect:
 key=(r['regime'],r['method'])
 actual=sum(groups[key])/len(groups[key]); historical=float(r['mean_rel_l2'])
 if len(groups[key])!=256 or abs(actual-historical)>5e-9:
  raise RuntimeError(f'Table-12 mismatch: {key}: n={len(groups[key])}, observed={actual:.10f}, expected={historical:.10f}')
 counts.append((key,actual))
from collections import defaultdict
byr=defaultdict(dict)
for (regime,operator,sample,method),v in value.items():byr[(regime,operator,sample)][method]=v
paired=[]
with (out/'PAIRWISE_STATS.csv').open(newline='') as f:prs=list(csv.DictReader(f))
try:
 from scipy.stats import wilcoxon
except ImportError:
 wilcoxon=None
for r in prs:
 reg,comp=r['regime'],r['comparator']
 ds=[v[comp]-v['IC-PRO-S'] for (rg,_,_),v in byr.items() if rg==reg]
 if len(ds)!=256:raise RuntimeError((reg,comp,'cases',len(ds)))
 diff=sum(ds)/len(ds)
 if abs(diff-float(r['comparator_minus_icpros_mean']))>5e-9:
  raise RuntimeError(f'Paired mean mismatch {reg}/{comp}: {diff}')
 if wilcoxon is not None:
  p=float(wilcoxon(ds,alternative='greater').pvalue)
  if abs(p-float(r['wilcoxon_one_sided_p']))>1e-7:
   raise RuntimeError(f'Wilcoxon p mismatch {reg}/{comp}: p={p}, original={r["wilcoxon_one_sided_p"]}')
 paired.append((reg,comp,diff))
print(json.dumps({'check':'PASS','a12_table12_means':len(counts),'a12_paired_differences':len(paired),
 'wilcoxon_recomputed':wilcoxon is not None,'case_rows':len(rows),
 'claims':'Independent CSV arithmetic replay, not original model re-training'},indent=2))
