"""Shortcut-audit diagnostics (2026-11-08): per-feature-block AUC, as-is vs histogram-equalized.
Run from the project root: python scripts/audit_diagnostics.py"""
import csv, numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from cxrf.audit import features, audit_auc
root=Path('data'); rng=np.random.default_rng(0)
inv={r['image_id']:r for r in csv.DictReader(open(root/'meta/inventory_raw.csv'))}
qc=[r for r in csv.DictReader(open(root/'meta/preprocess_qc.csv')) if not r['error'] and float(r['dyn_range'])>=0.15]
def load(src,n=1500):
    rows=[r for r in qc if r['source']==src]; rows=[rows[i] for i in sorted(rng.choice(len(rows),min(n,len(rows)),replace=False))]
    return np.stack([np.asarray(Image.open(root/r['out256'])) for r in rows]), np.array([inv[r['image_id']]['group_id'] for r in rows])
D={s:load(s) for s in ['nih','openi','shenzhen','synthchex']}
def eq(imgs): return np.stack([np.asarray(ImageOps.equalize(Image.fromarray(i))) for i in imgs])
for tag,tf in [('as-is',lambda x:x),('equalized',eq)]:
    F={s:features(tf(v[0])) for s,v in D.items()}
    for a,b in [('nih','synthchex'),('openi','synthchex'),('shenzhen','synthchex'),('nih','openi')]:
        y=np.r_[np.zeros(len(D[a][1])),np.ones(len(D[b][1]))].astype(int); g=np.r_[D[a][1],D[b][1]]
        res={blk: round(audit_auc(np.r_[F[a][blk],F[b][blk]],y,g,n_boot=50)['auc'],3) for blk in ['hist','stats','fft_low','marker']}
        allb=np.c_[tuple(np.r_[F[a][k],F[b][k]] for k in ['hist','stats','fft_low','marker'])]
        res['all']=round(audit_auc(allb,y,g,n_boot=50)['auc'],3)
        print(tag,a,'vs',b,res,flush=True)
