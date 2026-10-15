from __future__ import annotations
import csv,json,math,random,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.synthetic_v2 import make_wavelength_grid,make_instrument_bank,random_spectra,measure,batch_for_instrument
from src.core import FiRNO,tikhonov,rel_l2,sam,forward_residual

OUT=Path("/kaggle/working/N05V7");OUT.mkdir(parents=True,exist_ok=True)
D=torch.device("cuda");BANK_SEED=77;STD_NOISE=0.01

def fixed(split,bank,wl,seed,ns=160,noise=STD_NOISE):
    st=torch.random.get_rng_state();torch.manual_seed(seed);arr=[]
    for R0 in bank[split]:
        x=random_spectra(ns,wl,family="random");y=measure(R0,x,noise)
        arr.append((batch_for_instrument(R0,x),y,x))
    torch.random.set_rng_state(st);return arr

@torch.no_grad()
def ev(m,data,noise=STD_NOISE):
    m.eval();mr=[];tr=[];ms=[];ts=[];mf=[];tf=[];gm=[];etas=[]
    for R,y,x in data:
        o=m(R,y,noise);p=o["x"];t=tikhonov(R,y,1e-3)
        mr.append(rel_l2(p,x));tr.append(rel_l2(t,x));ms.append(sam(p,x));ts.append(sam(t,x))
        mf.append(forward_residual(R,p,y));tf.append(forward_residual(R,t,y))
        gm.append(o["gates"].mean(-1));etas.append(o["eta"])
    return {"model_rel":float(torch.cat(mr).mean()),"tik_rel":float(torch.cat(tr).mean()),
      "model_sam":float(torch.cat(ms).mean()),"tik_sam":float(torch.cat(ts).mean()),
      "model_fwd":float(torch.cat(mf).mean()),"tik_fwd":float(torch.cat(tf).mean()),
      "gate_mean":float(torch.cat(gm).mean()),"eta_mean":float(torch.cat(etas).mean())}

def train(seed,cfg,tag,steps=950,noise_random=True):
    torch.manual_seed(seed);random.seed(seed)
    wl=make_wavelength_grid(256,device=D);bank=make_instrument_bank(wl,seed=BANK_SEED)
    val=fixed("val",bank,wl,10001,96)
    m=FiRNO(**cfg).to(D);opt=torch.optim.AdamW(m.parameters(),lr=7e-4,weight_decay=1e-5)
    best=1e9;bad=0;hist=[];ck=OUT/f"{tag}_seed{seed}.pt"
    for step in range(1,steps+1):
        m.train();R0=random.choice(bank["train"]);x=random_spectra(32,wl,family="random")
        nl=random.uniform(0.004,0.020) if noise_random else STD_NOISE
        y=measure(R0,x,nl);R=batch_for_instrument(R0,x);o=m(R,y,nl)
        loss=rel_l2(o["x"],x).mean()+0.015*sam(o["x"],x).mean()+0.001*(o["raw"]**2).mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        if step%100==0:
            v=ev(m,val)["model_rel"];hist.append({"tag":tag,"seed":seed,"step":step,"val_rel":v})
            print(f"{tag} seed={seed} step={step} val={v:.5f}",flush=True)
            if v<best:
                best=v;bad=0;torch.save({"model":m.state_dict(),"cfg":cfg,"seed":seed,"best":best,"step":step},ck)
            else:bad+=1
            if step>=650 and bad>=3:break
    return ck,hist

def load(path):
    d=torch.load(path,map_location="cpu",weights_only=False);m=FiRNO(**d["cfg"]).to(D);m.load_state_dict(d["model"]);m.eval();return m,d

def stress(m):
    rows=[];wl=make_wavelength_grid(256,device=D);bank=make_instrument_bank(wl,seed=BANK_SEED)
    for nl in [0.0,0.005,0.01,0.02,0.05]:
        e=ev(m,fixed("test",bank,wl,5000+int(nl*10000),96,nl),nl)
        rows.append({"stress":"noise","level":nl,**e})
    base=fixed("test",bank,wl,6000,96)
    for frac in [0.25,0.5]:
        arr=[]
        for R,y,x in base:
            M=R.shape[1];keep=max(2,int(round(M*(1-frac))))
            idx=torch.linspace(0,M-1,keep,device=D).round().long().unique()
            arr.append((R[:,idx,:],y[:,idx],x))
        rows.append({"stress":"channel_dropout","level":frac,**ev(m,arr)})
    for sh in [1,3,6]:
        arr=[(torch.roll(R,sh,-1),y,x) for R,y,x in base]
        rows.append({"stress":"response_shift","level":sh,**ev(m,arr)})
    return rows

def resolution(m):
    rows=[]
    for n in [192,301,384]:
        wl=make_wavelength_grid(n,device=D);bank=make_instrument_bank(wl,seed=BANK_SEED)
        e=ev(m,fixed("test",bank,wl,7000+n,96))
        rows.append({"n":n,**e,"gain_pct":100*(e["tik_rel"]-e["model_rel"])/e["tik_rel"]})
    return rows

def lorentzian_batch(bs,N,device):
    # Matches the public Rec_Spectrometer synthetic prior: 1-3 Lorentzian peaks,
    # widths 10-25, magnitudes .6-1.4, centers in first 205 bins.
    xgrid=torch.arange(N,device=device).float();out=[]
    for _ in range(bs):
        k=random.randint(1,3);centers=[]
        tries=0
        while len(centers)<k and tries<200:
            c=random.randrange(0,min(205,N));tries+=1
            if all(abs(c-q)>=40 for q in centers):centers.append(c)
        while len(centers)<k:centers.append(random.randrange(0,min(205,N)))
        x=torch.zeros(N,device=device)
        for c in centers:
            w=random.uniform(10,25);mag=random.uniform(.6,1.4)
            q=w/(math.pi*((xgrid-c)**2+w*w));q=q/q.max();x=x+mag*q
        x=x-x.min();x=x/(x.max()+1e-8);out.append(x)
    return torch.stack(out)

def external_real(m):
    rep={"status":"FAILED"}
    repo=OUT/"Rec_Spectrometer"
    try:
        if repo.exists():shutil.rmtree(repo)
        subprocess.run(["git","clone","--depth","1","https://github.com/j1goblue/Rec_Spectrometer.git",str(repo)],
                       check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
        Rnp=np.genfromtxt(repo/"response_matrix.txt",dtype=np.float32);N=Rnp.shape[1]
        scale=float(np.max(np.abs(Rnp)));Rm=torch.tensor(N*Rnp/scale,device=D)
        # response-only adaptation: NO real spectra/measurements are used here.
        adapted=FiRNO(**torch.load(REP_CK,map_location="cpu",weights_only=False)["cfg"]).to(D)
        adapted.load_state_dict(torch.load(REP_CK,map_location="cpu",weights_only=False)["model"])
        adapted.train()
        opt=torch.optim.AdamW(adapted.parameters(),lr=3e-4,weight_decay=1e-5)
        for step in range(1,2201):
            x=lorentzian_batch(64,N,D)
            # Device-informed augmentation similar in spirit to the public Sim2Real generator.
            Rnoise=Rm*(1+0.05*torch.randn_like(Rm))
            Rb=Rnoise.unsqueeze(0).expand(x.shape[0],-1,-1)
            y=torch.einsum("bmn,bn->bm",Rb/N,x)
            y=y+1e-5*torch.relu(torch.randn_like(y))
            o=adapted(Rb,y,0.01)
            loss=rel_l2(o["x"],x).mean()+0.015*sam(o["x"],x).mean()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(adapted.parameters(),1.0);opt.step()
            if step%500==0:print(f"external response-only adaptation step={step} loss={float(loss):.5f}",flush=True)
        adapted.eval()

        # Load real data, using labels ONLY after adaptation for evaluation.
        names=[];ys=[];xs=[]
        ipdir=repo/"data/real_data_1/Ip";spdir=repo/"data/real_data_1/Spec"
        for sp in sorted(spdir.glob("*.txt")):
            ip=ipdir/f"Ip_{sp.name}"
            if not ip.exists():continue
            y=np.genfromtxt(ip,dtype=np.float32).reshape(-1)
            raw=np.genfromtxt(sp,dtype=np.float32);eff=np.arange(400,400+N)
            x=np.interp(eff,raw[:,0],raw[:,1]);x=x-x.min();x=x/(x.max()+1e-12)
            names.append(sp.name);ys.append(y);xs.append(x)
        Y=torch.tensor(np.stack(ys)/scale,device=D);X=torch.tensor(np.stack(xs),device=D)
        Rb=Rm.unsqueeze(0).expand(len(xs),-1,-1).contiguous()
        # Unknown source amplitude nuisance: estimate from physics-only Tikhonov,
        # calibrate measurement scale, then compare normalized spectral shapes.
        xr=tikhonov(Rb,Y,1e-3);amp=xr.amax(-1).clamp_min(1e-12);Yc=Y/amp[:,None]
        with torch.no_grad():
            p0=m(Rb,Yc,0.01)["x"];pa=adapted(Rb,Yc,0.01)["x"];tt=tikhonov(Rb,Yc,1e-3)
        def norm(q):return q/(q.amax(-1,keepdim=True)+1e-12)
        p0=norm(p0);pa=norm(pa);tt=norm(tt)

        # Official public Sim2Real checkpoint.
        sys.path.insert(0,str(repo))
        from models.model_log_sig import RecSpectrometer
        off=RecSpectrometer(16,N).cpu()
        off.load_state_dict(torch.load(repo/"saved_models/model_log_pk205_sig_iter20000.pth",map_location="cpu",weights_only=False))
        off.eval()
        Yraw=torch.tensor(np.stack(ys),dtype=torch.float32)
        ly=torch.log(torch.clamp(Yraw,min=1e-15));Yn=(ly-ly.min(1,keepdim=True).values)/(ly.max(1,keepdim=True).values-ly.min(1,keepdim=True).values+1e-15)
        with torch.no_grad():po=off(Yn);po=po/(po.amax(-1,keepdim=True)+1e-12)
        Xc=X.cpu()
        def metrics(q):
            qc=q.cpu()
            return {"rel_l2":float(rel_l2(qc,Xc).mean()),"sam":float(sam(qc,Xc).mean()),
                    "rmse":float(torch.sqrt(((qc-Xc)**2).mean(-1)).mean())}
        rep={"status":"OK","n_samples":len(xs),"num_sensors":int(Rnp.shape[0]),"num_wavelengths":N,
             "universal_zero_shot":metrics(p0),"response_only_adapted":metrics(pa),
             "tikhonov_shape_calibrated":metrics(tt),"official_sim2real_checkpoint":metrics(po),
             "real_labels_used_for_adaptation":False,"filenames":names}
        # single-peak wavelength MAE
        singles=[i for i,nm in enumerate(names) if nm.endswith("nm.txt")]
        wavelengths=np.arange(400,400+N)
        def pmae(q):
            q=q.cpu().numpy();xx=Xc.numpy()
            return float(np.mean([abs(wavelengths[np.argmax(q[i])]-wavelengths[np.argmax(xx[i])]) for i in singles]))
        rep["single_peak_mae_nm"]={"universal_zero_shot":pmae(p0),"response_only_adapted":pmae(pa),
                                   "tikhonov":pmae(tt),"official_sim2real":pmae(po)}
        torch.save({"model":adapted.state_dict(),"cfg":torch.load(REP_CK,map_location="cpu",weights_only=False)["cfg"]},
                   OUT/"FIRNO_EXTERNAL_RESPONSE_ONLY_ADAPTED.pt")
    except Exception as e:
        rep={"status":"FAILED_NONFATAL","error":repr(e)}
    return rep

def stats(seed_results):
    import scipy.stats as st
    gains=np.array([r["test_gain_pct"] for r in seed_results])
    return {"mean_test_gain_pct":float(gains.mean()),"std_test_gain_pct":float(gains.std(ddof=1)),
            "min_test_gain_pct":float(gains.min())}

def figures(seed_results,stress_rows,res_rows):
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(7,4));ax=fig.add_subplot(111)
    ax.plot([r["seed"] for r in seed_results],[r["test_model"] for r in seed_results],marker="o",label="FiRNO")
    ax.plot([r["seed"] for r in seed_results],[r["test_tikhonov"] for r in seed_results],marker="o",label="Tikhonov")
    ax.set_xlabel("Seed");ax.set_ylabel("Relative L2");ax.legend();fig.tight_layout();fig.savefig(OUT/"FIG_multiseed.png",dpi=220);plt.close(fig)
    n=[r for r in stress_rows if r["stress"]=="noise"]
    fig=plt.figure(figsize=(7,4));ax=fig.add_subplot(111)
    ax.plot([r["level"] for r in n],[r["model_rel"] for r in n],marker="o",label="FiRNO")
    ax.plot([r["level"] for r in n],[r["tik_rel"] for r in n],marker="o",label="Tikhonov")
    ax.set_xlabel("Noise std");ax.set_ylabel("Relative L2");ax.legend();fig.tight_layout();fig.savefig(OUT/"FIG_noise.png",dpi=220);plt.close(fig)

def main():
    global REP_CK
    # Stage 1: architecture selection on fixed validation.
    candidates={
      "RAW_RESIDUAL":{"width":64,"mode":"raw","prox":False,"use_fisher_channel":True},
      "FIRNO_GATE":{"width":64,"mode":"fisher_gate","prox":False,"use_fisher_channel":True},
      "FIRNO_GATE_PROX":{"width":64,"mode":"fisher_gate","prox":True,"use_fisher_channel":True},
      "HARD_NULL":{"width":64,"mode":"hard_null","prox":False,"use_fisher_channel":True},
    }
    sel={};history=[]
    for name,cfg in candidates.items():
        ck,h=train(7,cfg,"SEL_"+name,750);history+=h
        sel[name]=torch.load(ck,map_location="cpu",weights_only=False)["best"]
    winner=min(sel,key=sel.get)
    winner_cfg=candidates[winner]
    print("ARCHITECTURE_WINNER",winner,sel,flush=True)

    # Stage 2: 5 seeds for winner.
    wl=make_wavelength_grid(256,device=D);bank=make_instrument_bank(wl,seed=BANK_SEED)
    test=fixed("test",bank,wl,21001,192);ood=fixed("ood",bank,wl,31001,192)
    seed_results=[];cks=[]
    for seed in [3,7,11,19,23]:
        ck,h=train(seed,winner_cfg,"FINAL_"+winner,1050);history+=h;cks.append(ck)
        m,meta=load(ck);et=ev(m,test);eo=ev(m,ood)
        seed_results.append({"seed":seed,"step":meta["step"],"test_model":et["model_rel"],"test_tikhonov":et["tik_rel"],
          "test_gain_pct":100*(et["tik_rel"]-et["model_rel"])/et["tik_rel"],
          "ood_model":eo["model_rel"],"ood_tikhonov":eo["tik_rel"],
          "ood_gain_pct":100*(eo["tik_rel"]-eo["model_rel"])/eo["tik_rel"],
          "gate_mean":et["gate_mean"],"eta_mean":et["eta_mean"],
          "model_forward_residual":et["model_fwd"],"tikhonov_forward_residual":et["tik_fwd"]})
    # representative = median validation checkpoint
    cks=sorted(cks,key=lambda p:torch.load(p,map_location="cpu",weights_only=False)["best"])
    REP_CK=cks[len(cks)//2]
    repm,_=load(REP_CK);shutil.copy2(REP_CK,OUT/"FIRNO_REPRESENTATIVE_MODEL.pt")

    stress_rows=stress(repm);res_rows=resolution(repm)
    real=external_real(repm)

    report={"status":"V7_FIRNO_COMPLETE","architecture_selection":sel,"winner":winner,
      "winner_cfg":winner_cfg,"main_seed_results":seed_results,"statistics":stats(seed_results),
      "stress_tests":stress_rows,"resolution_transfer":res_rows,"external_real":real,
      "manuscript_gate":{
        "winner_is_fisher_gated":winner.startswith("FIRNO"),
        "all_5_test_positive":all(r["test_gain_pct"]>0 for r in seed_results),
        "all_5_ood_positive":all(r["ood_gain_pct"]>0 for r in seed_results),
        "mean_test_gain_ge_10pct":np.mean([r["test_gain_pct"] for r in seed_results])>=10,
        "external_real_ok":real.get("status")=="OK",
        "adapted_beats_official_sim2real_rmse": real.get("status")=="OK" and real["response_only_adapted"]["rmse"]<real["official_sim2real_checkpoint"]["rmse"]
      }}
    (OUT/"V7_FIRNO_REPORT.json").write_text(json.dumps(report,indent=2))
    with (OUT/"seed_results.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=seed_results[0].keys());w.writeheader();w.writerows(seed_results)
    with (OUT/"selection.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["candidate","validation_rel_l2"]);w.writerows(sel.items())
    with (OUT/"stress.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=stress_rows[0].keys());w.writeheader();w.writerows(stress_rows)
    with (OUT/"resolution.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=res_rows[0].keys());w.writeheader();w.writerows(res_rows)
    with (OUT/"training_history.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=history[0].keys());w.writeheader();w.writerows(history)
    figures(seed_results,stress_rows,res_rows)
    print(json.dumps(report,indent=2));print("V7_FIRNO_OK")

if __name__=="__main__":main()
