"""
Raman food analysis on REAL open data (Mendeley ctgg7k4m5g, edible oils).
Demonstrates the same pipeline the Orange .ows encodes:
  raw -> ALS baseline (fluorescence removal) -> cut fingerprint -> SNV -> PCA
       -> olive-vs-non-olive authentication (SVM/RF) + PLS peroxide-value regression.
Outputs PNG figures + results.json. Reproducible.
"""
import sys, codecs, json
if sys.platform.startswith('win'):
    sys.stdout=codecs.getwriter('utf-8')(sys.stdout.detach()); sys.stderr=codecs.getwriter('utf-8')(sys.stderr.detach())
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMG = ROOT / "figures"; IMG.mkdir(exist_ok=True)
DATA = ROOT / "data" / "Raman2.csv"
TEAL="#0284c7"; CORAL="#e11d48"; GOLD="#d97706"; GREEN="#16a34a"; INK="#0f172a"; GRID="#e2e8f0"
plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white",
    "axes.edgecolor":"#94a3b8","font.size":12,"axes.titlesize":14,"font.family":"DejaVu Sans",
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.unicode_minus":False})

OIL = {1:"EV Olive",2:"Extra-Light Olive",3:"Pure Olive",4:"Avocado",5:"Peanut",6:"Corn",
       7:"Grapeseed",8:"Safflower",9:"Hazelnut",10:"Flaxseed",11:"Almond",12:"Canola",
       13:"Avo/Flax/Olive blend",14:"Sesame",15:"Canola/Veg blend",16:"Vegetable",
       17:"Canola/Sun/Soy blend",18:"Sunflower",19:"Walnut"}

# ---- load ----
df = pd.read_csv(DATA)
wl = np.array([float(c) for c in df.columns[2:]])
cls = df["Class"].astype(int).values
pv = pd.to_numeric(df["PeroxideValue"], errors="coerce").values
X = df.iloc[:,2:].values.astype(float)
ok = ~np.isnan(X).any(axis=1)
X, cls, pv = X[ok], cls[ok], pv[ok]
res = {"n_samples":int(X.shape[0]), "n_bands_raw":int(X.shape[1]),
       "wn_min":round(float(wl.min()),0), "wn_max":round(float(wl.max()),0),
       "n_oil_types":int(len(np.unique(cls)))}
print(f"Loaded {X.shape[0]} samples x {X.shape[1]} bands ({wl.min():.0f}-{wl.max():.0f} cm-1), {len(np.unique(cls))} oil types")

# ---- ALS baseline (Eilers & Boelens) — fluorescence background removal ----
def als_baseline(y, lam=1e5, p=0.01, niter=10):
    L=len(y); D=sparse.diags([1,-2,1],[0,-1,-2],shape=(L,L-2)); D=lam*D.dot(D.transpose())
    w=np.ones(L); W=sparse.spdiags(w,0,L,L)
    for _ in range(niter):
        W.setdiag(w); Z=W+D; z=spsolve(Z,w*y); w=p*(y>z)+(1-p)*(y<z)
    return z
def baseline_correct(M):
    return np.vstack([row-als_baseline(row) for row in M])

Xbc = baseline_correct(X)
# cut to fingerprint region 400-1800 cm-1
fp = (wl>=400)&(wl<=1800); wlf=wl[fp]; Xf=Xbc[:,fp]
res["n_bands_fingerprint"]=int(fp.sum())
def snv(a): return (a-a.mean(1,keepdims=True))/a.std(1,keepdims=True)
Xs = snv(Xf)

# ============ FIG 1: raw vs baseline-corrected (fluorescence removal) ============
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
ex = np.where(cls==1)[0][:1]  # one EVOO sample
axes[0].plot(wl, X[ex[0]], color=CORAL, lw=1.5, label="raw (with fluorescence background)")
axes[0].plot(wl, als_baseline(X[ex[0]]), color=INK, lw=1.5, ls="--", label="ALS baseline (fluorescence)")
axes[0].set_title("Raman: raw spectrum has a fluorescence background", weight="bold")
axes[0].set_xlabel("Raman shift (cm⁻¹)"); axes[0].set_ylabel("Intensity"); axes[0].legend(fontsize=10)
for c,col in [(1,CORAL),(6,TEAL),(18,GOLD),(8,GREEN)]:
    m=cls==c
    if m.sum(): axes[1].plot(wlf, snv(Xf[m]).mean(0), color=col, lw=1.6, label=OIL[c])
axes[1].set_title("After baseline + cut(400–1800) + SNV: clean fingerprints", weight="bold")
axes[1].set_xlabel("Raman shift (cm⁻¹)"); axes[1].set_ylabel("SNV intensity"); axes[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig(IMG/"raman_fig1_baseline.png",dpi=150); plt.close(); print("[OK] fig1 baseline")

# ============ FIG 2: PCA ============
pca=PCA(n_components=10).fit(Xs); sc=pca.transform(Xs); ev=pca.explained_variance_ratio_
res["pca_pc1"]=round(float(ev[0]*100),1); res["pca_pc2"]=round(float(ev[1]*100),1)
res["pca_pc12"]=round(float(ev[:2].sum()*100),1)
is_olive = np.isin(cls,[1,2,3])
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
axes[0].bar(range(1,11),ev*100,color=TEAL,alpha=0.8); axes[0].plot(range(1,11),np.cumsum(ev)*100,color=CORAL,marker="o",lw=2)
axes[0].set_title("PCA explained variance",weight="bold"); axes[0].set_xlabel("PC"); axes[0].set_ylabel("Variance (%)")
axes[1].scatter(sc[is_olive,0],sc[is_olive,1],s=22,color=GREEN,alpha=0.7,label="Olive oils (class 1-3)",edgecolors="none")
axes[1].scatter(sc[~is_olive,0],sc[~is_olive,1],s=22,color=CORAL,alpha=0.5,label="Other / adulterant oils",edgecolors="none")
axes[1].set_title("PCA: olive vs other oils",weight="bold")
axes[1].set_xlabel(f"PC1 ({res['pca_pc1']}%)"); axes[1].set_ylabel(f"PC2 ({res['pca_pc2']}%)"); axes[1].legend(fontsize=10)
plt.tight_layout(); plt.savefig(IMG/"raman_fig2_pca.png",dpi=150); plt.close(); print(f"[OK] fig2 pca PC1={res['pca_pc1']} PC2={res['pca_pc2']}")

# ============ FIG 3: olive authentication (binary SVM/RF) ============
y = np.where(is_olive,"Olive","Other")
Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.3,random_state=42,stratify=y)
svm=make_pipeline(StandardScaler(),SVC(kernel="rbf",C=10,gamma="scale")).fit(Xtr,ytr)
preds=svm.predict(Xte); acc=accuracy_score(yte,preds)
rf=RandomForestClassifier(n_estimators=400,random_state=42).fit(Xtr,ytr)
res["svm_acc"]=round(float(acc*100),1)
res["rf_acc"]=round(float(accuracy_score(yte,rf.predict(Xte))*100),1)
res["svm_cv5"]=round(float(cross_val_score(make_pipeline(StandardScaler(),SVC(kernel='rbf',C=10,gamma='scale')),Xs,y,cv=5).mean()*100),1)
res["n_test"]=int(len(yte)); res["n_olive"]=int(is_olive.sum()); res["n_other"]=int((~is_olive).sum())
cm=confusion_matrix(yte,preds,labels=["Olive","Other"])
fig,ax=plt.subplots(figsize=(5.8,5.2)); ax.imshow(cm,cmap="Blues")
ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(["Olive","Other"]); ax.set_yticklabels(["Olive","Other"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Olive-oil authentication (SVM)\ntest acc={res['svm_acc']}%, 5-fold CV={res['svm_cv5']}%",weight="bold")
for i in range(2):
    for j in range(2): ax.text(j,i,cm[i,j],ha="center",va="center",fontsize=18,weight="bold",color="white" if cm[i,j]>cm.max()/2 else INK)
plt.tight_layout(); plt.savefig(IMG/"raman_fig3_authentication.png",dpi=150); plt.close(); print(f"[OK] fig3 auth SVM={res['svm_acc']}% CV={res['svm_cv5']}%")

# ============ FIG 4: PLS peroxide-value regression — confounding lesson ============
# CSV is ordered by class -> must use SHUFFLED CV folds.
from sklearn.model_selection import KFold
cv = KFold(5, shuffle=True, random_state=1)
def pls_cv(mask, ncomp):
    Xr, yr = Xs[mask], pv[mask]
    yp = cross_val_predict(PLSRegression(n_components=ncomp), Xr, yr, cv=cv).ravel()
    return yr, yp, r2_score(yr,yp), np.sqrt(mean_squared_error(yr,yp))
yr_all, yp_all, r2_all, rmse_all = pls_cv(np.ones(len(cls),bool), 10)
evoo = cls==1
yr_ev, yp_ev, r2_ev, rmse_ev = pls_cv(evoo, 8)
res.update({"pls_all_r2":round(float(r2_all),3),"pls_all_rmse":round(float(rmse_all),2),
    "pls_evoo_r2":round(float(r2_ev),3),"pls_evoo_rmse":round(float(rmse_ev),2),
    "pls_evoo_n":int(evoo.sum()),"pls_ncomp_all":10,"pls_ncomp_evoo":8})
fig,axes=plt.subplots(1,2,figsize=(11.4,5.2))
for ax,(yr,yp,r2,rmse,ttl,col) in zip(axes,[
        (yr_all,yp_all,r2_all,rmse_all,f"All 15 oil types (n={len(cls)})",CORAL),
        (yr_ev,yp_ev,r2_ev,rmse_ev,f"EV Olive Oil only (n={evoo.sum()})",GREEN)]):
    ax.scatter(yr,yp,s=24,color=col,alpha=0.6,edgecolors="none")
    lim=[min(yr.min(),yp.min())-3,max(yr.max(),yp.max())+3]; ax.plot(lim,lim,"--",color=INK,lw=1.2,alpha=.7)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f"{ttl}\nR²={r2:.3f}, RMSE={rmse:.2f} meq/kg",weight="bold")
    ax.set_xlabel("Actual peroxide value (meq/kg)"); ax.set_ylabel("Predicted")
fig.suptitle("PLS predicts oxidation (peroxide value) — but only after controlling for oil type",fontsize=13,weight="bold",y=1.02)
plt.tight_layout(); plt.savefig(IMG/"raman_fig4_peroxide.png",dpi=150,bbox_inches="tight"); plt.close()
print(f"[OK] fig4 PLS all R2={r2_all:.3f} | EVOO R2={r2_ev:.3f}")

json.dump(res, open(ROOT/"results.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n=== RESULTS ==="); print(json.dumps(res,ensure_ascii=False,indent=2))
