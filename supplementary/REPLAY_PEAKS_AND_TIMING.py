"""Independent retrospective diagnostic of archived IC-PRO-S A12 spectra.
Run from inside the expanded Zenodo evidence package with: 
 python supplementary/REPLAY_PEAKS_AND_TIMING.py
Requirements: torch, numpy, pandas, scipy.
A12 seed 7 and preserved exact test tensors; no original OOD guard is executed.
The synthetic peak count is a MORPHOLOGICAL diagnostic, not the generator family label.
"""
from __future__ import annotations
import json, os, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.signal import find_peaks, peak_widths
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'a12_frozen_source'))
from src.core import FiRNO,tikhonov,rel_l2
OUT=Path(__file__).resolve().parent/'results'; OUT.mkdir(parents=True,exist_ok=True)
try: torch.set_num_threads(1)
except Exception: pass
if not (BASE/'frozen_inputs/A12_EXACT_BENCHMARK_TENSORS.pt').exists():
 raise RuntimeError('Requires full Zenodo evidence package, not GitHub source only.')
ck=torch.load(BASE/'a12_frozen_source/checkpoints/FINAL_FIRNO_GATE_PROX_seed7.pt', map_location='cpu',weights_only=False)
data=torch.load(BASE/'frozen_inputs/A12_EXACT_BENCHMARK_TENSORS.pt',map_location='cpu',weights_only=False)
model=FiRNO(**ck['cfg']).to('cpu').eval();model.load_state_dict(ck['model'])
archive=pd.read_csv(BASE/'a12_frozen_outputs/sample_errors_core.csv')

def features(x,p):
    # Indices are spectral bins. No device-specific wavelength calibration is implied.
    x=np.asarray(x, float);p=np.asarray(p,float)
    x_m=float(np.max(x));p_m=float(np.max(p));
    gt_ks,_=find_peaks(x,prominence=max(0.10*x_m,1e-10),distance=4)
    pred_ks,_=find_peaks(p,prominence=max(0.10*p_m,1e-10),distance=4)
    bin_gt=int(np.argmax(x));bin_pred=int(np.argmax(p))
    area_err=float(abs(p.sum()-x.sum())/max(x.sum(),1e-8))
    h_err=float(abs(p_m-x_m)/max(x_m,1e-8))
    # Compute FWHM only for genuinely interior prominent main maxima; otherwise undefined.
    def mainwidth(v,k,peaks):
        if k not in peaks: return np.nan
        return float(peak_widths(v,np.array([k]),rel_height=0.5)[0][0])
    wx=mainwidth(x,bin_gt,gt_ks); wp=mainwidth(p,bin_pred,pred_ks)
    width_error=float(abs(wp-wx)) if np.isfinite(wx) and np.isfinite(wp) else np.nan
    # Recall of resolvable 10%-prominence local peaks within ±5 spectral bins.
    hits=sum(bool(len(pred_ks) and np.min(np.abs(pred_ks-k))<=5) for k in gt_ks)
    peak_recall=float(hits/len(gt_ks)) if len(gt_ks) else np.nan
    morphology=('no_interior_peak' if not len(gt_ks) else 'single_interior_peak' if len(gt_ks)==1 else 'multiple_interior_peaks')
    return dict(morphology=morphology, n_gt_peaks=len(gt_ks), n_pred_peaks=len(pred_ks),main_peak_bin_error=abs(bin_pred-bin_gt), main_peak_height_relative_error=h_err, integrated_area_relative_error=area_err, main_peak_half_prominence_width_abs_error_bins=width_error, peak_recall_tol5_bins=peak_recall)

rows=[];fail=[]
with torch.no_grad():
 for regime,arr in data.items():
  for record in arr:
   op=record['name'];RR=record['R'];xs=record['test_x']; ys=record['test_y'];
   for offset in range(0,len(xs),16):
    x=xs[offset:offset+16];y=ys[offset:offset+16];R=RR.unsqueeze(0).expand(len(x),-1,-1)
    pred=model(R,y,noise_std=0.01)['x'];errs=rel_l2(pred,x).numpy()
    for i in range(len(x)):
     idx=offset+i
     r={'regime':regime,'operator':op,'sample':idx,'seed':7,'rel_l2':float(errs[i]),**features(x[i].numpy(),pred[i].numpy())}
     rows.append(r)
# verify exact archived spectrum-level error for every one of 768 reconstructions.
df=pd.DataFrame(rows)
historical=archive.loc[(archive.method=='IC-PRO-S')&(archive.seed==7),['operator','sample','rel_l2']]
merged=df.merge(historical,on=['operator','sample'],how='left',validate='one_to_one',suffixes=('_replayed','_archived'))
max_diff=float(abs(merged.rel_l2_replayed-merged.rel_l2_archived).max())
if len(df)!=768 or not np.isfinite(max_diff) or max_diff>2e-5:
 raise RuntimeError(f'Archived replay not equal to historic seed7 cases: count={len(df)} maxdiff={max_diff}')
df.to_csv(OUT/'A12_SEED7_PER_CASE_SPECTRAL_FEATURES.csv',index=False)
metrics=['rel_l2','main_peak_bin_error','main_peak_height_relative_error','integrated_area_relative_error','main_peak_half_prominence_width_abs_error_bins','peak_recall_tol5_bins']
def aggregate(d,grp):
 g=d.groupby(grp,dropna=False)
 x=g[metrics].agg(['mean','median','count']);x.columns=['_'.join(t) for t in x.columns];return x.reset_index()
aggregate(df,['regime','morphology']).to_csv(OUT/'A12_MORPHOLOGY_STRATIFIED_METRICS.csv',index=False)
aggregate(df,['regime']).to_csv(OUT/'A12_REGIME_SPECTRAL_METRICS.csv',index=False)
worst=df.sort_values('rel_l2',ascending=False).groupby('regime').head(3)
worst.to_csv(OUT/'A12_WORST_NINE_CASES.csv',index=False)
# Measured-source A13B case-level *precomputed* global-maximum peak metrics, not recalculation from shape.
per=BASE/'a13b/outputs/PER_CASE.csv'
if per.exists():
 d=pd.read_csv(per)
 q=d[(d.method=='IC-PRO-S response-only') & (d.measurement_condition=='controlled_noiseless')]
 grp=q.groupby(['M','group']).agg(n=('case','nunique'),mean_peak_abs_nm=('peak_abs_nm','mean'),median_peak_abs_nm=('peak_abs_nm','median'),mean_rel_l2=('rel_l2','mean')).reset_index()
 grp.to_csv(OUT/'A13B_ARCHIVED_PEAK_LOCALIZATION_BY_GROUP.csv',index=False)

# A reproducible, hardware-specific, one-thread CPU microbenchmark of *reconstruction only*.
# The archival OOD guard is not available; time it as NA, not zero.
rec=data['interpolation'][0]
R=rec['R'].unsqueeze(0).contiguous(); y=rec['test_y'][0:1].contiguous();N=R.shape[-1]

def profile_once():
 d={};t=time.perf_counter()
 x0=tikhonov(R,y,1e-3);d['anchor_solve_ms']=(time.perf_counter()-t)*1000;t=time.perf_counter()
 A=R/N;bp=torch.einsum('bmn,bm->bn',A,y);bp=bp/(bp.abs().amax(-1,keepdim=True)+1e-6)
 grad=torch.zeros_like(x0);grad[:,1:-1]=0.5*(x0[:,2:]-x0[:,:-2]);fmap=model.fisher_map(R,0.01)
 f=[x0,bp,grad];
 if model.use_fisher_channel:f.append(fmap)
 feats=torch.stack(f,1);d['response_preprocess_ms']=(time.perf_counter()-t)*1000;t=time.perf_counter()
 h=model.inp(feats)
 for b in model.blocks:h=b(h)
 raw=torch.tanh(model.out(h).squeeze(1))*(0.60*torch.sigmoid(model.step))
 d['neural_proposal_ms']=(time.perf_counter()-t)*1000;t=time.perf_counter()
 dp,gates,S,dnull,drow=model.split_and_gate(R,y,x0,raw,0.01);z=x0+dp
 d['svd_mode_gating_ms']=(time.perf_counter()-t)*1000;t=time.perf_counter()
 if model.prox:z,eta=model.proximal(z,R,y,S,gates,0.01)
 d['proximal_solve_ms']=(time.perf_counter()-t)*1000;t=time.perf_counter()
 prediction=z.clamp_min(0.0)
 d['nonnegativity_projection_ms']=(time.perf_counter()-t)*1000
 return prediction,d
with torch.inference_mode():
 for _ in range(20):
  probe,tms=profile_once();baseline=model(R,y,0.01)['x']
 if not torch.allclose(probe,baseline,rtol=1e-5,atol=1e-6):raise RuntimeError('Stage-split inference mismatches full model.')
 profiles=[];full=[]
 for _ in range(120):
  _,tms=profile_once();profiles.append(tms)
  t=time.perf_counter();model(R,y,0.01)['x'];full.append((time.perf_counter()-t)*1000)
pt=pd.DataFrame(profiles)
records=[dict(stage=k,median_ms=float(pt[k].median()),p25_ms=float(pt[k].quantile(.25)),p75_ms=float(pt[k].quantile(.75))) for k in pt.columns]
records.append(dict(stage='complete_reconstruction_only',median_ms=float(np.median(full)),p25_ms=float(np.percentile(full,25)),p75_ms=float(np.percentile(full,75))))
records.append(dict(stage='historical_OOD_guard',median_ms=np.nan,p25_ms=np.nan,p75_ms=np.nan))
records.append(dict(stage='full_selective_routing',median_ms=np.nan,p25_ms=np.nan,p75_ms=np.nan))
pd.DataFrame(records).to_csv(OUT/'CPU_RECONSTRUCTION_STAGE_TIMING.csv',index=False)
info={'test':'single-thread CPU timing and frozen A12 seed-7 spectral diagnostics','n_cases':len(df),'operators':df.operator.nunique(),'seed':7,'input_source':'frozen_inputs/A12_EXACT_BENCHMARK_TENSORS.pt','checkpoint':'a12_frozen_source/checkpoints/FINAL_FIRNO_GATE_PROX_seed7.pt','max_abs_rel_l2_difference_vs_archive':max_diff,'cpu': next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),platform.processor()),'torch_version':torch.__version__,'threads':torch.get_num_threads(),'batch_size':1,'warmup':20,'timed_repetitions':120,'factorization_cache':'not used','OOD_guard':'not timed; historical implementation unavailable','note':'Diagnostic on one CPU / one frozen M16 sensing matrix; cannot compare directly with historical Table 10 timings from a different environment. No added wet-lab or independently acquired spectrometry data.'}
(OUT/'RUN_CONFIGURATION.json').write_text(json.dumps(info,indent=2,default=str),encoding='utf-8')
print(json.dumps(info,indent=2,default=str));print(pd.DataFrame(records).to_string(index=False))
print('A12 regime peak breakdown:'); print(aggregate(df,['regime'])[[ 'regime','main_peak_bin_error_mean','main_peak_height_relative_error_mean','integrated_area_relative_error_mean','main_peak_half_prominence_width_abs_error_bins_mean','peak_recall_tol5_bins_mean']].to_string(index=False))
