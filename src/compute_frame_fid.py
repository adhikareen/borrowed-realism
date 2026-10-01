import sys, argparse, glob, json
from pathlib import Path
sys.path.insert(0,"src")
import numpy as np, torch
import torch.nn.functional as F
from compute_fvd import frechet_distance

def load_inception(dev):
    from pytorch_fid.inception import InceptionV3
    m = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()
    return m

ap=argparse.ArgumentParser()
ap.add_argument("--gen-dir",required=True); ap.add_argument("--real-dir",default="runs/real_decode_F17_val")
ap.add_argument("--out-json",required=True); ap.add_argument("--n",type=int,default=2000)
ap.add_argument("--gpu",type=int,default=0); ap.add_argument("--bs",type=int,default=16)
a=ap.parse_args()
dev=torch.device(f"cuda:{a.gpu}")
FR=[0,5,11,16]
def feats(d,cachef):
    p=Path("runs/fid_feats")/f"{cachef}.npy"
    if p.exists(): return np.load(p)
    p.parent.mkdir(exist_ok=True)
    fs=(sorted(glob.glob(f"{d}/gen_*.npy")) or sorted(glob.glob(f"{d}/real_*.npy")))[:a.n]
    net=load_inception(dev); out=[]
    for i in range(0,len(fs),a.bs):
        ims=[]
        for f_ in fs[i:i+a.bs]:
            arr=np.load(f_).astype(np.float32)
            for k in FR: ims.append(arr[:,k])
        x=torch.from_numpy(np.stack(ims)).to(dev).clamp(0,1)
        x=F.interpolate(x,size=(299,299),mode="bilinear",align_corners=False)
        with torch.no_grad(): f2=net(x)[0].squeeze(-1).squeeze(-1)
        out.append(f2.cpu())
    r=torch.cat(out).numpy(); np.save(p,r); return r
fr=feats(a.real_dir,"real_"+Path(a.real_dir).name)
fg=feats(a.gen_dir,"gen_"+Path(a.gen_dir).name)
fid=float(frechet_distance(fg,fr))
Path(a.out_json).write_text(json.dumps({"fid":round(fid,3),"gen":a.gen_dir,"frames_per_clip":len(FR)}))
print("FID",Path(a.gen_dir).name,round(fid,2))
