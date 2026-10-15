from __future__ import annotations
import os,sys,json,csv,random,time,traceback,hashlib,zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.core import FiRNO,tikhonov,rel_l2,sam,forward_residual
from src.synthetic_v2 import make_wavelength_grid,make_instrument_bank
from benchmark import build,save_manifest
from baselines import nnls_tv_batch,train_pnp_denoiser,pnp_reconstruct,train_mgdun,train_respec
from pigdm import PiGDM

OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True);CK=OUT/'checkpoints';CK.mkdir(exist_ok=True)
QUICK=os.environ.get('N05_QUICK','0')=='1'
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not QUICK and DEVICE.type!='cuda': raise RuntimeError('FULL run requires GPU. Use Kaggle GPU or set N05_QUICK=1 for smoke test.')


def _jsonable(x):
    if isinstance(x,dict): return {str(k):_jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [_jsonable(v) for v in x]
    if torch.is_tensor(x): return x.detach().cpu().tolist() if x.numel()!=1 else x.item()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    return str(x)

def save_status(stage,status='OK',extra=None):
    p=OUT/'RUN_STATUS.json';d=json.loads(p.read_text()) if p.exists() else {};d[stage]={'status':status,'time':time.strftime('%Y-%m-%d %H:%M:%S'),**_jsonable(extra or {})};p.write_text(json.dumps(d,indent=2),encoding='utf-8')

def done(s):return (OUT/f'.done_{s}').exists()
def mark(s):(OUT/f'.done_{s}').write_text('OK')

def snapshot(label='partial'):
    """Atomically write a resumable state archive after a stage.
    ZIP_STORED is intentional: model tensors compress poorly and speed matters.
    If interrupted while writing, the previous valid snapshot is preserved.
    """
    zpath=Path('/kaggle/working/N05_A12_PARTIAL_STATE.zip') if Path('/kaggle/working').exists() else ROOT.parent/'N05_A12_PARTIAL_STATE.zip'
    tmp=zpath.with_suffix('.tmp.zip')
    if tmp.exists(): tmp.unlink()
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_STORED) as z:
        for p in OUT.rglob('*'):
            if p.is_file(): z.write(p,p.relative_to(ROOT))
    os.replace(tmp,zpath)
    print('PARTIAL_SNAPSHOT',label,zpath,flush=True)
    return zpath

def flatten(data):
    return [(reg,d) for reg,items in data.items() for d in items]

def batchR(R,B):return R.unsqueeze(0).expand(B,-1,-1)

def err(p,x):return rel_l2(p,x).detach().cpu().numpy()

def append_rows(path,rows):
    path=Path(path);first=not path.exists();
    with path.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());
        if first:w.writeheader()
        w.writerows(rows)

def load_ic(seed):
    p=ROOT/'checkpoints'/f'FINAL_FIRNO_GATE_PROX_seed{seed}.pt';d=torch.load(p,map_location='cpu',weights_only=False);m=FiRNO(**d['cfg']).to(DEVICE);m.load_state_dict(d['model']);m.eval();return m,d


def stage_ic_classical(data):
    path=OUT/'sample_errors_core.csv'
    if done('core'):return
    rows=[]
    # NNLS lambda validation-only across all extension validation operators
    lambdas=[1e-4,3e-4,1e-3,3e-3,1e-2] if not QUICK else [1e-3]
    scores={}
    for lam in lambdas:
        vv=[]
        for d in original_val_items(): vv.append(err(nnls_tv_batch(d['R'],d['val_y'],lam=lam,iters=120 if not QUICK else 3),d['val_x']))
        scores[lam]=float(np.concatenate(vv).mean())
    bestlam=min(scores,key=scores.get);(OUT/'NNLS_TUNING.json').write_text(json.dumps({'grid':scores,'selected_lambda':bestlam},indent=2))
    models={s:load_ic(s)[0] for s in [3,7,11,19,23]}
    with torch.no_grad():
      for reg,d in flatten(data):
        x,y,R=d['test_x'],d['test_y'],d['R'];B=x.shape[0];Rb=batchR(R,B)
        tp=tikhonov(Rb,y,1e-3);te=err(tp,x)
        for i,e in enumerate(te):rows.append({'regime':reg,'operator':d['name'],'sample':i,'method':'Tikhonov','seed':'','rel_l2':float(e)})
        npred=nnls_tv_batch(R,y,lam=bestlam,iters=120 if not QUICK else 3);ne=err(npred,x)
        for i,e in enumerate(ne):rows.append({'regime':reg,'operator':d['name'],'sample':i,'method':'NNLS-TV','seed':'','rel_l2':float(e)})
        for s,m in models.items():
            p=m(Rb,y,.01)['x'];ee=err(p,x)
            for i,e in enumerate(ee):rows.append({'regime':reg,'operator':d['name'],'sample':i,'method':'IC-PRO-S','seed':s,'rel_l2':float(e)})
    append_rows(path,rows);save_status('core','OK',{'nnls_lambda':bestlam});mark('core')


def stage_pnp(data):
    if done('pnp'):return
    m,meta=train_pnp_denoiser(CK,DEVICE,QUICK);rows=[]
    with torch.no_grad():
      for reg,d in flatten(data):
        p=pnp_reconstruct(m,d['R'],d['test_y'],steps=12 if not QUICK else 2,eta=.65,ridge=1e-3);ee=err(p,d['test_x'])
        for i,e in enumerate(ee):rows.append({'regime':reg,'operator':d['name'],'sample':i,'method':'PnP tuned','seed':'','rel_l2':float(e)})
    append_rows(OUT/'sample_errors_extra.csv',rows);save_status('pnp','OK',{'best_val_mse':meta.get('best_val_mse'),'step':meta.get('step')});mark('pnp')


def original_train_bank():
    wl=make_wavelength_grid(256,device=DEVICE);return make_instrument_bank(wl,seed=77)['train']

def original_val_items():
    from src.synthetic_v2 import random_spectra,measure
    wl=make_wavelength_grid(256,device=DEVICE);bank=make_instrument_bank(wl,seed=77);items=[]
    st=torch.random.get_rng_state();torch.manual_seed(10001)
    for j,R in enumerate(bank['val']):
        x=random_spectra(24 if QUICK else 96,wl,family='random');y=measure(R,x,.01);items.append({'name':f'orig_val_{j}','R':R,'val_x':x,'val_y':y})
    torch.random.set_rng_state(st);return items

def stage_mgdun(data):
    if done('mgdun'):return
    vals=original_val_items();rows=[];met=[]
    for s in [3,7,11]:
        m,meta=train_mgdun(s,original_train_bank(),vals,CK,DEVICE,QUICK);met.append(meta)
        with torch.no_grad():
          for reg,d in flatten(data):
            p=m(d['R'],d['test_y']);ee=err(p,d['test_x'])
            for i,e in enumerate(ee):rows.append({'regime':reg,'operator':d['name'],'sample':i,'method':'MGDUN-style','seed':s,'rel_l2':float(e)})
    append_rows(OUT/'sample_errors_extra.csv',rows);save_status('mgdun','OK',{'seeds':[3,7,11]});mark('mgdun')

def stage_respec(data):
    if done('respec'):return
    rows=[];meta={}
    for reg,d in flatten(data):
        m,md=train_respec(d,CK,DEVICE,QUICK);meta[d['name']]={'best':md['best'],'step':md['step']}
        with torch.no_grad():ee=err(m(d['test_y']),d['test_x'])
        for i,e in enumerate(ee):rows.append({'regime':reg,'operator':d['name'],'sample':i,'method':'ReSpecNN per-device','seed':'','rel_l2':float(e)})
    append_rows(OUT/'sample_errors_extra.csv',rows);save_status('respec','OK',meta);mark('respec')

def stage_pigdm(data):
    if done('pigdm'):return
    pg=PiGDM(DEVICE);meta=pg.train_or_load(CK,QUICK);rows=[]
    # fixed stochastic seed so initial q is reproducible
    torch.manual_seed(20268)
    for reg,d in flatten(data):
        # batches to control VRAM
        es=[];bs=16 if not QUICK else 2
        for j in range(0,d['test_x'].shape[0],bs):
            p=pg.reconstruct(d['R'],d['test_y'][j:j+bs],gamma=.05,K=50 if not QUICK else 3);es.extend(err(p,d['test_x'][j:j+bs]).tolist())
        for i,e in enumerate(es):rows.append({'regime':reg,'operator':d['name'],'sample':i,'method':'PiGDM controlled','seed':'','rel_l2':float(e)})
    append_rows(OUT/'sample_errors_extra.csv',rows);save_status('pigdm','OK',{'selected_step':meta.get('step'),'params':sum(p.numel() for p in pg.model.parameters())});mark('pigdm')


def aggregate(data):
    import pandas as pd
    from scipy.stats import wilcoxon
    dfs=[]
    for p in [OUT/'sample_errors_core.csv',OUT/'sample_errors_extra.csv']:
        if p.exists():dfs.append(pd.read_csv(p))
    df=pd.concat(dfs,ignore_index=True)
    # primary table: IC seed7, single comparators, MGDUN averaged per sample across seeds
    primary=df[(df.method!='IC-PRO-S') | (df.seed.fillna(-1).astype(float)==7)].copy()
    mg=primary[primary.method=='MGDUN-style'].groupby(['regime','operator','sample'],as_index=False).rel_l2.mean();mg['method']='MGDUN-style';mg['seed']='mean3'
    primary=pd.concat([primary[primary.method!='MGDUN-style'],mg],ignore_index=True)
    summary=[];stats=[]
    deploy={
      'IC-PRO-S':('No','one shared deterministic family model'), 'Tikhonov':('No','analytic per response'), 'NNLS-TV':('No','120-iteration optimization'),
      'PnP tuned':('No','shared denoiser + 12 iterative steps'),'MGDUN-style':('No','shared six-stage unfolding model'),'PiGDM controlled':('No','shared diffusion prior + 50 guided reverse steps'),
      'ReSpecNN per-device':('Yes: per target response','separate target-response neural model')}
    for reg in primary.regime.unique():
        rr=primary[primary.regime==reg];tik=rr[rr.method=='Tikhonov'][['operator','sample','rel_l2']].rename(columns={'rel_l2':'tik'})
        for method in rr.method.unique():
            mm=rr[rr.method==method];vals=mm.rel_l2.values;joined=mm.merge(tik,on=['operator','sample'])
            frac=float((joined.rel_l2<joined.tik).mean()*100) if method!='Tikhonov' else np.nan
            summary.append({'regime':reg,'method':method,'mean_rel_l2':float(vals.mean()),'std_rel_l2':float(vals.std(ddof=1)) if len(vals)>1 else 0,
                            'fraction_better_than_tikhonov_pct':frac,'target_retraining':deploy[method][0],'deployment':deploy[method][1]})
            if method not in ('Tikhonov','IC-PRO-S'):
                ic=rr[rr.method=='IC-PRO-S'][['operator','sample','rel_l2']].rename(columns={'rel_l2':'ic'});z=mm.merge(ic,on=['operator','sample'])
                diff=z.rel_l2-z.ic; rng=np.random.default_rng(2026);boots=[]
                for _ in range(2000 if not QUICK else 20):
                    idx=rng.integers(0,len(diff),len(diff));boots.append(float(diff.iloc[idx].mean()))
                ci=np.quantile(boots,[.025,.975]);
                try:p=float(wilcoxon(diff,alternative='greater').pvalue)
                except:p=float('nan')
                stats.append({'regime':reg,'comparator':method,'comparator_minus_icpros_mean':float(diff.mean()),'bootstrap95_lo':float(ci[0]),'bootstrap95_hi':float(ci[1]),'wilcoxon_one_sided_p':p})
    pd.DataFrame(summary).to_csv(OUT/'EXTENDED_TABLE6.csv',index=False);pd.DataFrame(stats).to_csv(OUT/'PAIRWISE_STATS.csv',index=False)
    # Per-operator table keeps M=8 and M=40 separate instead of hiding them in a regime mean.
    byop=[]
    for (reg,op,method),g in primary.groupby(['regime','operator','method']):
        vals=g.rel_l2.values
        tg=primary[(primary.regime==reg)&(primary.operator==op)&(primary.method=='Tikhonov')][['sample','rel_l2']].rename(columns={'rel_l2':'tik'})
        j=g.merge(tg,on='sample')
        frac=float((j.rel_l2<j.tik).mean()*100) if method!='Tikhonov' else np.nan
        byop.append({'regime':reg,'operator':op,'M':int(next(d['R'].shape[0] for rr,d in flatten(data) if rr==reg and d['name']==op)),
                     'method':method,'mean_rel_l2':float(vals.mean()),'std_rel_l2':float(vals.std(ddof=1)) if len(vals)>1 else 0,
                     'fraction_better_than_tikhonov_pct':frac,'target_retraining':deploy[method][0],'deployment':deploy[method][1]})
    pd.DataFrame(byop).to_csv(OUT/'EXTENDED_TABLE6_BY_OPERATOR.csv',index=False)
    # five-seed IC-PRO stability table
    ic=df[df.method=='IC-PRO-S'].groupby(['regime','seed'],as_index=False).rel_l2.mean();ic.to_csv(OUT/'ICPRO_5SEED_EXTRAPOLATION.csv',index=False)
    # integrity / completion
    required=['IC-PRO-S','Tikhonov','NNLS-TV','PnP tuned','MGDUN-style','ReSpecNN per-device','PiGDM controlled'];seen=set(primary.method.unique());missing=sorted(set(required)-seen)
    final={'status':'PASS' if not missing else 'INCOMPLETE','quick_mode':QUICK,'device':str(DEVICE),'methods':sorted(seen),'missing':missing,
           'primary_rows':int(len(primary)),'all_error_rows':int(len(df))}
    (OUT/'FINAL_STATUS.json').write_text(json.dumps(final,indent=2));
    # result zip
    zpath=Path('/kaggle/working/N05_A12_CROSS_INSTRUMENT_RESULTS.zip') if Path('/kaggle/working').exists() else ROOT.parent/'N05_A12_CROSS_INSTRUMENT_RESULTS.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file() and not p.name.startswith('.done_'):z.write(p,p.relative_to(ROOT))
    print(json.dumps(final,indent=2));print('RESULT_ZIP',zpath);return final

def main():
    print('='*72);print('N05 A12 STRONG-Q1 CROSS-INSTRUMENT BASELINE EXPANSION');print('device=',DEVICE,'quick=',QUICK);print('='*72)
    data=build(DEVICE,QUICK);save_manifest(data,OUT/'BENCHMARK_MANIFEST.json')
    stages=[('core',stage_ic_classical),('pnp',stage_pnp),('mgdun',stage_mgdun),('respec',stage_respec),('pigdm',stage_pigdm)]
    for name,fn in stages:
        print('\n### STAGE',name,flush=True)
        try:
            fn(data); snapshot(name)
        except Exception as e:
            save_status(name,'FAILED',{'error':repr(e),'traceback':traceback.format_exc()});snapshot(name+'_FAILED');print(traceback.format_exc());raise
    aggregate(data);print('A12_COMPLETE_OK')
if __name__=='__main__':main()
