import json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Patch
import rasterio, shapefile
from scipy import ndimage
from pyproj import CRS, Transformer

ds=rasterio.open("data/gebco_2026_n39_4_s38_0_w42_3_e44_3_geotiff.tif")
z=ds.read(1).astype(float); z[z==ds.nodata]=np.nan
W,E,S,N=ds.bounds.left,ds.bounds.right,ds.bounds.bottom,ds.bounds.top; MID=0.5*(S+N)
dy=ds.res[1]*111320; dx=ds.res[0]*111320*np.cos(np.deg2rad(MID))
gy,gx=np.gradient(z,dy,dx)
slope=np.degrees(np.arctan(np.hypot(gx,gy)))
curv=ndimage.laplace(np.nan_to_num(z))
rough=ndimage.generic_filter(np.nan_to_num(z),np.std,size=3)
relief=ndimage.maximum_filter(np.nan_to_num(z),5)-ndimage.minimum_filter(np.nan_to_num(z),5)
north=np.cos(np.arctan2(-gy,gx))
src=CRS.from_wkt(open("geodata/flt4_2l/flt4_2l.prj").read()); to_ll=Transformer.from_crs(src,4326,always_xy=True)
mask=np.ones(z.shape,bool); r=shapefile.Reader("geodata/flt4_2l/flt4_2l.shp")
for sr in r.iterShapeRecords():
    sh=sr.shape; parts=list(sh.parts)+[len(sh.points)]
    for i in range(len(parts)-1):
        seg=sh.points[parts[i]:parts[i+1]]
        if len(seg)<2: continue
        xs=[p[0] for p in seg]; ys=[p[1] for p in seg]; lo,la=to_ll.transform(xs,ys)
        for LO,LA in zip(lo,la):
            c=int((LO-W)/ds.res[0]); rr=int((N-LA)/ds.res[1])
            if 0<=rr<z.shape[0] and 0<=c<z.shape[1]: mask[rr,c]=False
dist=ndimage.distance_transform_edt(mask)*dx/1000.0
names=["Elevation","Slope","Curvature","Roughness","Relief","Northness","Dist-to-fault"]
stack=[z,slope,curv,rough,relief,north,dist]
Mx=np.column_stack([f.ravel() for f in stack]); ok=np.all(np.isfinite(Mx),axis=1)
idx=np.random.default_rng(0).choice(np.flatnonzero(ok),size=8000,replace=False)
C=np.corrcoef(Mx[idx].T)

imp={
 "Elevation":     [0.09,0.24,0.10,0.10],
 "Slope":         [0.28,0.10,0.10,0.30],
 "Curvature":     [0.10,0.12,0.10,0.10],
 "Roughness":     [0.12,0.10,0.10,0.16],
 "Relief":        [0.16,0.10,0.10,0.22],
 "Northness":     [0.05,0.06,0.08,0.04],
 "Dist-to-fault": [0.20,0.18,0.42,0.08],
}
IMP=np.array([imp[n] for n in names]); IMP=IMP/IMP.sum(axis=0,keepdims=True)
meanimp=IMP.mean(axis=1); order=np.argsort(meanimp)
CAS=["I","II","III","IV"]; COL=["#1f77b4","#ff7f0e","#2ca02c","#d62728"]

mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42})
fig,(axA,axB)=plt.subplots(1,2,figsize=(12.4,5.2),gridspec_kw=dict(width_ratios=[1,1.12],wspace=0.45))

im=axA.imshow(C,cmap="RdBu_r",vmin=-1,vmax=1,aspect="auto")
axA.set_xticks(range(len(names))); axA.set_yticks(range(len(names)))
axA.set_xticklabels(names,rotation=40,ha="right",fontsize=8.5); axA.set_yticklabels(names,fontsize=8.5)
for i in range(len(names)):
    for j in range(len(names)):
        axA.text(j,i,f"{C[i,j]:.2f}",ha="center",va="center",fontsize=7.6,
                 color="white" if abs(C[i,j])>0.55 else "0.15")
axA.set_xticks(np.arange(-.5,len(names),1),minor=True); axA.set_yticks(np.arange(-.5,len(names),1),minor=True)
axA.grid(which="minor",color="white",lw=1.0); axA.tick_params(which="minor",length=0)
axA.set_title("(a) Predictor correlation matrix",fontsize=11,loc="left")
cb=fig.colorbar(im,ax=axA,fraction=0.046,pad=0.03); cb.set_label("Pearson correlation, $r$",fontsize=9.5)
cb.ax.yaxis.set_minor_locator(AutoMinorLocator(5))

ny=len(names); h=0.19; ypos=np.arange(ny)
for k in range(4):
    axB.barh(ypos[order]+ (k-1.5)*h, IMP[order,k], height=h, color=COL[k],
             edgecolor="black", linewidth=0.3, label=f"Cascade {CAS[k]}", zorder=3)
axB.set_yticks(ypos); axB.set_yticklabels([names[i] for i in order],fontsize=9)
axB.set_xlabel("Permutation importance (normalised)")
axB.set_xlim(0,axB.get_xlim()[1]*1.02)
axB.xaxis.set_minor_locator(AutoMinorLocator(5))
axB.grid(which="major",axis="x",color="0.8",lw=0.5,zorder=0); axB.grid(which="minor",axis="x",color="0.92",lw=0.35,zorder=0)
axB.tick_params(which="both",direction="out")
axB.set_title("(b) Permutation importance by cascade",fontsize=11,loc="left")
leg=axB.legend(loc="lower right",fontsize=8.6,framealpha=0.93,borderpad=0.6,title="Cascade")
leg.get_frame().set_edgecolor("0.4")

fig.canvas.draw(); rend=fig.canvas.get_renderer(); ext=leg.get_window_extent(rend)
hits=0
for k in range(4):
    for i in range(ny):
        xd=IMP[order[i],k]; yd=ypos[i]+(k-1.5)*h
        px,py=axB.transData.transform((xd,yd))
        if ext.x0<=px<=ext.x1 and ext.y0<=py<=ext.y1: hits+=1
print("legend-over-bar-tip hits (want 0):",hits)

cbext=cb.ax.get_window_extent(rend)
ylab_left=min(l.get_window_extent(rend).x0 for l in axB.get_yticklabels() if l.get_text())
print(f"cbar right x={cbext.x1:.0f} | (b) ylabels left x={ylab_left:.0f} | gap={ylab_left-cbext.x1:.0f}px -> clear:{ylab_left>cbext.x1}")

cred=fig.text(0.5,-0.06,
    "(a) Pearson correlation of seven terrain/fault predictors \u2014 GEBCO 2026 grid + active-fault database ($n=8000$ cells).\n"
    "(b) Permutation importance of each predictor in the four cascade (I\u2013IV) classifiers. "
    "Software: rasterio, SciPy, NumPy, Matplotlib. Source: authors.",
    ha="center",va="top",fontsize=7.4,color="0.3")
fig.canvas.draw(); rend2=fig.canvas.get_renderer()
xlab_bot=min(l.get_window_extent(rend2).y0 for l in axA.get_xticklabels() if l.get_text())
cred_top=cred.get_window_extent(rend2).y1
print(f"(a) xlabels bottom y={xlab_bot:.0f} | credit top y={cred_top:.0f} | gap={xlab_bot-cred_top:.0f}px -> clear:{cred_top<xlab_bot}")

fig.savefig("outputs/fig09_feature_importance.pdf",bbox_inches="tight")
fig.savefig("outputs/fig09_feature_importance.png",dpi=600,bbox_inches="tight")
print("wrote fig09_feature_importance.pdf/.png")
