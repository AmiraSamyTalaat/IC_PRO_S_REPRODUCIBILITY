#!/usr/bin/env python3
"""Verify repeated spectral inputs under paired A12 response operators.

Applies only to A12 preserved *exact benchmark inputs*. Assumes no broader
training set identity, external measured-device independence or OOD guarantees.
"""
from pathlib import Path
import argparse,json,hashlib,torch

def digest(x):return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
 d=torch.load(a.input,map_location='cpu',weights_only=True)
 rec=[]
 for rg,v in d.items():
  if len(v)!=2:raise AssertionError(f'Expected 2 operators in {rg}')
  a1,a2=v
  for split,rows in [('test_x',128),('val_x',64)]:
   x1,x2=a1[split],a2[split]
   if x1.shape!=(rows,256) or x2.shape!=(rows,256):raise AssertionError((rg,split,x1.shape,x2.shape))
   if not torch.equal(x1,x2):raise AssertionError(f'Within-regime {split} is not shared! {rg}')
   exact_unique=len({digest(row) for row in x1})
   if exact_unique!=rows:raise AssertionError(f'Exact duplicates WITHIN the {split} {rg}: {exact_unique} of {rows}')
   rec.append({'regime':rg,'split':split,'response_operators':[a1['name'],a2['name']],
             'observations':rows*2,'unique_input_spectra':rows,
             'spectra_identical_for_two_operators':True,'unique_spectra_sha256_count':exact_unique,
             'sha256_array':digest(x1),'operator_matrices_identical':bool(torch.equal(a1['R'],a2['R']))})
  ta={digest(z) for z in a1['test_x']};va={digest(z) for z in a1['val_x']}
  if ta & va:raise AssertionError(f'Validation and test share exact spectrum in {rg}: {len(ta & va)}')
 summary={'status':'PASS','source':'Exact archived A12 input tensors','records':rec,
 'no_exact_val_test_spectrum_overlap_within_regime':True,
 'important_scope':'Exact equality audit only; near-duplicate shapes and training/test leakage not excluded',
 'statistical_unit':'128 unique spectra per regime; two repeated-operator observations for each'}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(summary,indent=2),encoding='utf-8')
 print(json.dumps({'status':'PASS','regimes':len(d),'test_evaluations':3*2*128,'distinct_test_spectra_within_regimes':3*128},indent=2))
if __name__=='__main__':main()
