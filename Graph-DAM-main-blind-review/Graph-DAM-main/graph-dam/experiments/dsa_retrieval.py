"""UCI Daily and Sports Activities (DSA) retrieval experiment for Graph-DAM.

Protocol matches the repository experiments: N=30 memories, 20% corruption,
25 log-spaced beta values from 1e-1 to 1e3, 5 retrieval steps, 5 runs.
Each DSA memory is one real 5-second subject/activity segment represented as a
45-node body-sensor correlation graph (5 locations x 9 sensor channels).
"""
import os, glob, re, shutil, zipfile, urllib.request, warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

SEED=42; EPS=1e-12; N_MEM=30; N_RUNS=5; N_STEPS=5
BETAS=np.logspace(-1,3,25); N_BODY=5; CH_PER_BODY=9; N_NODES=45
KNN=8; EDGE_REMOVE_FRAC=.20; RAW_REMOVE_FRAC=.20
URLS=["https://archive.ics.uci.edu/static/public/256/daily+and+sports+activities.zip",
      "https://archive.ics.uci.edu/ml/machine-learning-databases/00256/data.zip"]
ZIP="/tmp/dsa.zip"; ROOT="/tmp/dsa_graphdam"
ACT={1:"Sitting",2:"Standing",3:"Lying on back",4:"Lying on right side",5:"Ascending stairs",6:"Descending stairs",7:"Standing in elevator",8:"Moving in elevator",9:"Walking",10:"Walking treadmill",11:"Walking incline",12:"Running",13:"Stepper",14:"Cross trainer",15:"Cycling horizontal",16:"Cycling vertical",17:"Rowing",18:"Jumping",19:"Basketball"}

def valid_zip(p): return os.path.exists(p) and os.path.getsize(p)>100000 and zipfile.is_zipfile(p)
if not valid_zip(ZIP):
    for u in URLS:
        try:
            req=urllib.request.Request(u,headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req,timeout=180) as r: data=r.read()
            open(ZIP,"wb").write(data)
            if valid_zip(ZIP): break
        except Exception: pass
if not valid_zip(ZIP): raise RuntimeError("Could not download DSA dataset")
if os.path.exists(ROOT): shutil.rmtree(ROOT)
os.makedirs(ROOT); zipfile.ZipFile(ZIP).extractall(ROOT)
for _ in range(3):
    changed=False
    for zp in glob.glob(ROOT+"/**/*.zip",recursive=True):
        dest=zp+"_extract"
        if os.path.exists(dest): continue
        try: os.makedirs(dest); zipfile.ZipFile(zp).extractall(dest); changed=True
        except Exception: pass
    if not changed: break

def ids(path):
    p=path.replace("\\","/").lower(); a=re.search(r"/a(\d+)",p); s=re.search(r"/p(\d+)",p)
    if not a or not s: return None
    nums=re.findall(r"\d+",os.path.basename(p)); return int(s.group(1)),int(a.group(1)),int(nums[-1]) if nums else 0
records=[]
for p in glob.glob(ROOT+"/**/*.txt",recursive=True):
    z=ids(p)
    if z and 1<=z[0]<=8 and 1<=z[1]<=19: records.append((*z,p))
records.sort()

def load(p):
    try: Z=np.loadtxt(p,delimiter=",",dtype=float)
    except Exception: Z=np.loadtxt(p,dtype=float)
    if Z.ndim==1: Z=Z[None,:]
    return Z[:,:45]
def standardize(Z):
    sd=Z.std(0,keepdims=True); sd[sd<1e-8]=1
    return (Z-Z.mean(0,keepdims=True))/sd
def adjacency(Z):
    C=np.nan_to_num(np.abs(np.corrcoef(standardize(Z),rowvar=False))); np.fill_diagonal(C,0)
    A0=np.zeros_like(C)
    for i in range(N_NODES):
        j=np.argpartition(C[i],-KNN)[-KNN:]; A0[i,j]=C[i,j]
    A=np.maximum(A0,A0.T); np.fill_diagonal(A,0); return A
def lap(A):
    A=(A+A.T)/2; np.fill_diagonal(A,0); return np.diag(A.sum(1))-A
def rawfeat(Z):
    dx=np.diff(Z,axis=0)
    return np.concatenate([Z.mean(0),Z.std(0),np.sqrt(np.mean(Z**2,0)),np.median(Z,0),np.mean(np.abs(dx),0),np.ptp(Z,axis=0)])
def corrupt_raw(Z,rng):
    T=len(Z); rm=rng.choice(np.arange(1,T-1),size=min(int(.2*T),T-2),replace=False); keep=np.ones(T,bool); keep[rm]=False; t=np.arange(T); Q=np.empty_like(Z)
    for j in range(Z.shape[1]): Q[:,j]=np.interp(t,t[keep],Z[keep,j])
    return Q

groups={}
for s,a,k,p in records: groups.setdefault((s,a),[]).append((k,p))
cands=[]
for (s,a),fs in sorted(groups.items()):
    k,p=sorted(fs)[len(fs)//2]
    try:
        Z=load(p); A=adjacency(Z); cands.append(dict(subject=s,activity=a,path=p,signal=Z,A=A,L=lap(A),raw=rawfeat(Z)))
    except Exception: pass
if len(cands)<N_MEM: raise RuntimeError(f"Only {len(cands)} usable memories")

def opd(A,B): return np.max(np.abs(np.linalg.eigvalsh(A-B)))
CL=np.stack([x["L"] for x in cands]); center=CL.mean(0); first=int(np.argmax([opd(x,center) for x in CL])); sel=[first]
md=np.array([opd(x,CL[first]) for x in CL]); md[first]=-np.inf
while len(sel)<N_MEM:
    j=int(np.argmax(md)); sel.append(j)
    for i in range(len(CL)):
        if np.isfinite(md[i]): md[i]=min(md[i],opd(CL[i],CL[j]))
    md[j]=-np.inf
M=[cands[i] for i in sel]; MA=np.stack([x["A"] for x in M]); ML=np.stack([x["L"] for x in M]); R0=np.stack([x["raw"] for x in M])
rm=R0.mean(0,keepdims=True); rs=R0.std(0,keepdims=True); rs[rs<1e-8]=1; MR=(R0-rm)/rs
LF=ML.reshape(N_MEM,-1); target=np.arange(N_MEM)

def corrupt_edges(A,rng):
    Q=A.copy(); e=np.argwhere(np.triu(Q>0,1)); n=max(1,int(.2*len(e))); uv=e[rng.choice(len(e),n,replace=False)]; Q[uv[:,0],uv[:,1]]=0; Q[uv[:,1],uv[:,0]]=0; return Q
QL=[]; QR=[]; QSIG=[]
for run in range(N_RUNS):
    rg=np.random.default_rng(SEED+10000+run); rr=np.random.default_rng(SEED+20000+run); ql=[]; qr=[]; qs=[]
    for i in range(N_MEM):
        ql.append(lap(corrupt_edges(MA[i],rg))); z=corrupt_raw(M[i]["signal"],rr); qs.append(z); qr.append(((rawfeat(z)[None,:]-rm)/rs)[0])
    QL.append(np.stack(ql)); QR.append(np.stack(qr)); QSIG.append(qs)

def op_d2(Q):
    ev=np.linalg.eigvalsh(Q[:,None,:,:]-ML[None,:,:,:]); return np.max(np.abs(ev),axis=-1)**2
DMM=op_d2(ML); SCALE_OP=max(float(np.median(DMM[np.triu_indices(N_MEM,1)])),EPS)
S=LF@LF.T; SCALE_EU=max(float(np.median(np.std(S,axis=1))),EPS); S=MR@MR.T; SCALE_RAW=max(float(np.median(np.std(S,axis=1))),EPS)
def softmax(Z): Z=Z-Z.max(1,keepdims=True); E=np.exp(Z); return E/E.sum(1,keepdims=True)
def graph_retrieve(Q,b):
    Q=Q.copy(); pred=None
    for _ in range(N_STEPS): W=softmax(-b*op_d2(Q)/SCALE_OP); Q=(W@LF).reshape(-1,N_NODES,N_NODES); pred=np.argmax(W,1)
    return pred
def eu_retrieve(Q,b):
    Q=Q.reshape(len(Q),-1).copy(); pred=None
    for _ in range(N_STEPS): W=softmax(b*(Q@LF.T)/SCALE_EU); Q=W@LF; pred=np.argmax(Q@LF.T,1)
    return pred
def raw_retrieve(Q,b):
    Q=Q.copy(); pred=None
    for _ in range(N_STEPS): W=softmax(b*(Q@MR.T)/SCALE_RAW); Q=W@MR; pred=np.argmax(Q@MR.T,1)
    return pred
AG=np.zeros((N_RUNS,len(BETAS))); AE=AG.copy(); AR=AG.copy()
for r in range(N_RUNS):
    for j,b in enumerate(BETAS):
        AG[r,j]=np.mean(graph_retrieve(QL[r],b)==target); AE[r,j]=np.mean(eu_retrieve(QL[r],b)==target); AR[r,j]=np.mean(raw_retrieve(QR[r],b)==target)
    print(f"run {r+1}/{N_RUNS}")
def ms(X): return X.mean(0),X.std(0,ddof=1)/np.sqrt(len(X))
mg,sg=ms(AG); me,se=ms(AE); mr,sr=ms(AR)
os.makedirs("figures",exist_ok=True)
fig,ax=plt.subplots(figsize=(5.2,3.65))
for y,e,m,l in [(mg,sg,"o","Graph-DAM"),(me,se,"s","Eu-DAM"),(mr,sr,"^","Raw Eu-DAM")]: ax.errorbar(BETAS,y,yerr=e,marker=m,lw=1.4,capsize=2,label=l)
ax.set_xscale("log"); ax.set_ylim(0,1.03); ax.set_xlabel(r"Inverse temperature $\beta_n$"); ax.set_ylabel("Retrieval accuracy"); ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); fig.savefig("figures/dsa_retrieval_accuracy.png",dpi=300); fig.savefig("figures/dsa_retrieval_accuracy.pdf"); plt.close(fig)
# Representative signal + 3D figures
b=BETAS[np.argmax(mg)]; pg=graph_retrieve(QL[0],b); pe=eu_retrieve(QL[0],b); pr=raw_retrieve(QR[0],b)
c=np.where((pg==target)&(pe!=target)&(pr!=target))[0]
if not len(c): c=np.where((pg==target)&((pe!=target)|(pr!=target)))[0]
i=int(c[0]) if len(c) else 0; idx=[i,i,int(pg[i]),int(pe[i]),int(pr[i])]
def name(k): return f"S{M[k]['subject']} — {ACT[M[k]['activity']]}"
Z0=[QSIG[0][i],M[i]["signal"],M[idx[2]]["signal"],M[idx[3]]["signal"],M[idx[4]]["signal"]]; mu=Z0[1].mean(0,keepdims=True); sd=Z0[1].std(0,keepdims=True); sd[sd<1e-8]=1; Z=[(x-mu)/sd for x in Z0]
titles=["(b) Corrupted query","(c) Target memory","(d) Graph-DAM","(e) Eu-DAM","(f) Raw Eu-DAM"]; names=[name(k) for k in idx]; rows=[(0,"Torso"),(3,"Right leg"),(4,"Left leg")]
fig,axs=plt.subplots(3,5,figsize=(12,5.7),sharex=True,sharey=True)
for col,z in enumerate(Z):
    t=np.linspace(0,1,len(z))
    for row,(body,label) in enumerate(rows):
        ids=np.arange(body*9,body*9+3)
        for k in range(3): axs[row,col].plot(t,z[:,ids[k]],lw=.8)
        axs[row,col].axhline(0,color=".85",lw=.4); axs[row,col].spines[["top","right"]].set_visible(False)
for c0 in range(5): axs[0,c0].set_title(titles[c0]); axs[2,c0].set_xlabel("Normalized time"); axs[2,c0].text(.5,-.3,names[c0],transform=axs[2,c0].transAxes,ha="center",fontsize=7)
for r,(_,lab) in enumerate(rows): axs[r,0].set_ylabel(lab+"\nacceleration")
fig.tight_layout(); fig.savefig("figures/dsa_retrieval_signals.png",dpi=300); fig.savefig("figures/dsa_retrieval_signals.pdf"); plt.close(fig)
fig=plt.figure(figsize=(12,6.8))
for r,(body,lab) in enumerate(rows):
    ids=np.arange(body*9,body*9+3); vals=np.concatenate([z[:,ids].ravel() for z in Z]); lim=max(float(np.percentile(np.abs(vals),99)),1)
    for c0,z in enumerate(Z):
        ax=fig.add_subplot(3,5,r*5+c0+1,projection="3d"); x=z[:,ids]; ax.plot(x[:,0],x[:,1],x[:,2],lw=.8); ax.scatter(*x[0],s=12,marker="o"); ax.scatter(*x[-1],s=16,marker="x"); ax.set(xlim=(-lim,lim),ylim=(-lim,lim),zlim=(-lim,lim)); ax.view_init(22,-55)
        if r==0: ax.set_title(titles[c0])
        if r==2: ax.text2D(.5,-.1,names[c0],transform=ax.transAxes,ha="center",fontsize=7)
fig.tight_layout(); fig.savefig("figures/dsa_retrieval_3d.png",dpi=300); fig.savefig("figures/dsa_retrieval_3d.pdf"); plt.close(fig)
print("Saved DSA figures to figures/")
