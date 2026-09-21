import csv
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, MultipleLocator

W,E,S,N = 42.3,44.3,38.0,39.4
mags=[]; decyr=[]
for r in csv.DictReader(open("data/IEB_Turkey_5000_events.csv", newline="")):
    try:
        lo=float(r["Lon"]); la=float(r["Lat"]); m=float(r["Mag"])
        y=int(r["Year"]); mo=int(r["Month"]); d=int(r["Day"])
    except: continue
    if W<=lo<=E and S<=la<=N:
        mags.append(m); decyr.append(y+(mo-1)/12+(d-1)/365)
mags=np.array(mags); decyr=np.array(decyr)

dM=0.2
edges=np.arange(round(mags.min()/dM)*dM, mags.max()+dM, dM); centers=edges[:-1]+dM/2
inc,_=np.histogram(mags, bins=edges)
cum=np.array([(mags>=c-1e-9).sum() for c in centers])
Mc=centers[np.argmax(inc)]
sel=mags>=Mc-1e-9; Nmc=int(sel.sum())
b=np.log10(np.e)/(mags[sel].mean()-(Mc-dM/2)); b_err=b/np.sqrt(Nmc)
a=np.log10(Nmc)+b*Mc
print(f"N={len(mags)} Mc={Mc:.1f} b={b:.2f}+/-{b_err:.2f} a={a:.2f}")

mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,
                     "axes.linewidth":1.0})
BLUE="#1f77b4"; ORANGE="#ff7f0e"; RED="#d62728"; GREY="#555555"
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.9), gridspec_kw=dict(width_ratios=[1,1.28], wspace=0.30))

axA.scatter(centers, np.where(cum>0,cum,np.nan), s=42, facecolor=BLUE, edgecolor="black",
            linewidths=0.5, zorder=5, label="Cumulative")
axA.scatter(centers, np.where(inc>0,inc,np.nan), s=34, facecolor="none", edgecolor=ORANGE,
            linewidths=1.4, marker="s", zorder=4, label="Incremental")
mm=np.linspace(Mc, mags.max()+0.15, 50)
axA.plot(mm, 10**(a-b*mm), color="black", lw=1.8, zorder=6, label="G\u2013R fit")
axA.axvline(Mc, color=GREY, ls=(0,(5,3)), lw=1.2, zorder=3)
axA.text(Mc+0.05, 1.15, f"$M_c={Mc:.1f}$", color=GREY, fontsize=9.5, rotation=90,
         va="bottom", ha="left")
axA.set_yscale("log")
axA.set_xlim(3.9, 7.6); axA.set_ylim(0.7, 500)
axA.set_xlabel("Magnitude, $M$"); axA.set_ylabel("Number of earthquakes")
axA.xaxis.set_major_locator(MultipleLocator(0.5)); axA.xaxis.set_minor_locator(AutoMinorLocator(5))
axA.yaxis.set_major_locator(LogLocator(base=10)); axA.yaxis.set_minor_locator(LogLocator(base=10, subs="auto", numticks=12))
axA.grid(which="major", color="0.75", lw=0.5); axA.grid(which="minor", color="0.9", lw=0.35)
axA.tick_params(which="both", direction="in", top=True, right=True)

statA=axA.text(0.97, 0.95, f"$b = {b:.2f} \\pm {b_err:.2f}$\n$a = {a:.2f}$\n$M_c = {Mc:.1f}$\n$N = {len(mags)}$",
         transform=axA.transAxes, ha="right", va="top", fontsize=9.5,
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.5", lw=0.8))
axA.legend(loc="lower left", fontsize=8.4, framealpha=0.92, borderpad=0.6, handletextpad=0.5)
axA.set_title("(a) Frequency\u2013magnitude distribution", fontsize=11, loc="left")

order=np.argsort(decyr); t=decyr[order]; mg=mags[order]
axB.scatter(t, mg, s=10+6*(mg-4.0), facecolor=BLUE, edgecolor="black", linewidths=0.3,
            alpha=0.7, zorder=4, label="Earthquakes ($M$ vs time)")
axB.set_xlim(1968, 2028); axB.set_ylim(3.9, 7.7)
axB.set_xlabel("Year"); axB.set_ylabel("Magnitude, $M$")
axB.xaxis.set_major_locator(MultipleLocator(10)); axB.xaxis.set_minor_locator(AutoMinorLocator(5))
axB.yaxis.set_major_locator(MultipleLocator(1)); axB.yaxis.set_minor_locator(AutoMinorLocator(5))
axB.grid(which="major", color="0.75", lw=0.5); axB.grid(which="minor", color="0.9", lw=0.35)
axB.tick_params(which="both", direction="in", top=False, right=False)

axC=axB.twinx()
axC.step(np.sort(decyr), np.arange(1,len(decyr)+1), where="post", color=RED, lw=1.8,
         zorder=5, label="Cumulative count")
axC.set_ylim(0, len(decyr)*1.03); axC.set_ylabel("Cumulative number", color=RED)
axC.tick_params(axis="y", colors=RED, which="both", direction="in")
axC.yaxis.set_minor_locator(AutoMinorLocator(5))
axC.spines["right"].set_color(RED)

annos=[(1976.9,7.3,"1976  M7.3",(1989,6.55)),(2011.81,7.1,"2011 Van\nsequence",(2000,6.35))]
annoObjs=[]
for yr,mw,lab,(tx,ty) in annos:
    annoObjs.append(axB.annotate(lab, xy=(yr,mw), xytext=(tx,ty), fontsize=8.4, ha="center",
        va="top", fontweight="bold", arrowprops=dict(arrowstyle="->", color="0.25", lw=1.0), zorder=6))

h1,l1=axB.get_legend_handles_labels(); h2,l2=axC.get_legend_handles_labels()
axB.legend(h1+h2, l1+l2, loc="upper center", bbox_to_anchor=(0.5,0.99), fontsize=8.2,
           framealpha=0.93, borderpad=0.6, ncol=1)
axB.set_title("(b) Temporal seismicity pattern", fontsize=11, loc="left")
axB.set_zorder(axC.get_zorder()+1); axB.patch.set_visible(False)

fig.text(0.5, -0.02,
    "Data: IEB catalogue (study window, as in Fig. 6), 1970\u20132026. "
    "$b$-value by Aki (1965) maximum likelihood; $M_c$ by MAXC. Plotted with Matplotlib. Source: authors.",
    ha="center", va="top", fontsize=7.6, color="0.3")

fig.canvas.draw(); rend=fig.canvas.get_renderer()
def npin(ax, xs, ys, ext):
    d=ax.transData.transform(np.column_stack([xs,ys]))
    return int(np.sum((d[:,0]>=ext.x0)&(d[:,0]<=ext.x1)&(d[:,1]>=ext.y0)&(d[:,1]<=ext.y1)))
report=[]
lA=axA.get_legend().get_window_extent(rend)
report.append(("A legend / cumulative", npin(axA,centers,np.where(cum>0,cum,-1),lA)))
report.append(("A legend / incremental", npin(axA,centers,np.where(inc>0,inc,-1),lA)))
bA=statA.get_window_extent(rend)
report.append(("A statbox / cumulative", npin(axA,centers,np.where(cum>0,cum,-1),bA)))
report.append(("A statbox / incremental", npin(axA,centers,np.where(inc>0,inc,-1),bA)))
lB=axB.get_legend().get_window_extent(rend)
report.append(("B legend / mag-scatter", npin(axB,decyr,mags,lB)))
for k,o in enumerate(annoObjs):
    report.append((f"B anno{k} / mag-scatter", npin(axB,decyr,mags,o.get_window_extent(rend))))
print("OVERLAP CHECK (want 0):")
for name,n in report: print(f"  {name:28s}: {n}")
print("ALL CLEAR" if all(n==0 for _,n in report) else "!! OVERLAP PRESENT")

fig.savefig("outputs/fig07_gutenberg_richter.pdf", bbox_inches="tight")
fig.savefig("outputs/fig07_gutenberg_richter.png", dpi=600, bbox_inches="tight")
print("wrote fig07_gutenberg_richter.pdf/.png")
