import csv
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable as SM
from matplotlib.patches import Polygon, Ellipse
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import matplotlib.patheffects as pe
import rasterio, shapefile
from scipy import ndimage
from pyproj import CRS, Transformer

DEM="data/gebco_2026_n39_4_s38_0_w42_3_e44_3_geotiff.tif"
EQ ="data/IEB_Turkey_5000_events.csv"
GEO="geodata/geo4_2l/geo4_2l"

ds=rasterio.open(DEM); z=ds.read(1).astype(float); z[z==ds.nodata]=np.nan
W,E,S,N=ds.bounds.left,ds.bounds.right,ds.bounds.bottom,ds.bounds.top; MID=0.5*(S+N)
COSL=np.cos(np.deg2rad(MID)); dx=ds.res[0]*111320*COSL; dy=ds.res[1]*111320
gy,gx=np.gradient(z,dy,dx); slope=np.degrees(np.arctan(np.hypot(gx,gy)))
relief=ndimage.maximum_filter(np.nan_to_num(z),5)-ndimage.minimum_filter(np.nan_to_num(z),5)

CALD=np.array([(43.72,39.00),(43.90,39.09),(44.09,39.18),(44.30,39.27)])
lon1d=W+(np.arange(z.shape[1])+0.5)*ds.res[0]; lat1d=N-(np.arange(z.shape[0])+0.5)*ds.res[1]
LON,LAT=np.meshgrid(lon1d,lat1d)
def dist_to_polyline(LON,LAT,pts):
    d=np.full(LON.shape,1e9)
    for i in range(len(pts)-1):
        ax_,ay_=pts[i]; bx_,by_=pts[i+1]
        vx=(bx_-ax_)*COSL; vy=(by_-ay_); L2=vx*vx+vy*vy
        wx=(LON-ax_)*COSL; wy=(LAT-ay_)
        t=np.clip((wx*vx+wy*vy)/L2,0,1)
        px=wx-t*vx; py=wy-t*vy
        d=np.minimum(d,np.hypot(px,py)*111.0)
    return d
distf=dist_to_polyline(LON,LAT,CALD)

sn=np.clip(slope/35.0,0,1); rn=np.clip(relief/500.0,0,1); fp=np.exp(-distf/10.0)
Sraw=2.6*sn+1.8*fp+0.8*rn-2.3
Sus=1/(1+np.exp(-Sraw))
print("susceptibility min/med/max:",round(np.nanmin(Sus),2),round(np.nanmedian(Sus),2),round(np.nanmax(Sus),2))

hs=LightSource(315,45).hillshade(z,vert_exag=1.5,dx=dx,dy=dy)
grey=LinearSegmentedColormap.from_list("g",["#8f8f8f","#ffffff"])
src=CRS.from_wkt(open(GEO+".prj").read()); to_ll=Transformer.from_crs(src,4326,always_xy=True)
lake=[]; r=shapefile.Reader(GEO+".shp"); gi=[f[0] for f in r.fields[1:]].index("GLG")
for sr in r.iterShapeRecords():
    if sr.record[gi]!="H2O": continue
    sh=sr.shape; parts=list(sh.parts)+[len(sh.points)]
    for i in range(len(parts)-1):
        seg=sh.points[parts[i]:parts[i+1]]
        if len(seg)>=3:
            xs=[p[0] for p in seg]; ys=[p[1] for p in seg]; lo,la=to_ll.transform(xs,ys)
            lake.append(np.column_stack([lo,la]))
epi=None
for row in csv.DictReader(open(EQ,newline="")):
    try:
        if int(row["Year"])==1976 and float(row["Mag"])>=7.0:
            epi=(float(row["Lon"]),float(row["Lat"])); break
    except: pass

mpl.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42})
fig,ax=plt.subplots(figsize=(11.6,8.6))
ax.imshow(hs,extent=[W,E,S,N],origin="upper",cmap=grey,vmin=0,vmax=1,zorder=0,aspect="auto",interpolation="bilinear")

Sm=np.ma.masked_less(Sus,0.15)
sus_cmap=plt.cm.YlOrRd
im=ax.imshow(Sm,extent=[W,E,S,N],origin="upper",cmap=sus_cmap,vmin=0,vmax=1,alpha=0.65,zorder=1,aspect="auto",interpolation="bilinear")
if lake:
    ax.add_collection(PatchCollection([Polygon(a,closed=True) for a in lake],
        facecolor="#cfe3f0",edgecolor="#2f6d8f",linewidths=0.7,alpha=0.85,zorder=2))
halo=[pe.withStroke(linewidth=2.4,foreground="white")]

ax.plot(CALD[:,0],CALD[:,1],color="#5a1e6e",lw=2.6,zorder=4,solid_capstyle="round",
        path_effects=[pe.withStroke(linewidth=4.2,foreground="white")])
ax.text(44.18,39.18,"\u00c7ald\u0131ran F.",color="#5a1e6e",fontsize=8.6,fontweight="bold",
        rotation=27,ha="center",va="top",zorder=6,path_effects=halo)

ax.add_patch(Ellipse((43.95,39.03),0.72,0.44,angle=28,facecolor="none",edgecolor="#111",
        lw=1.8,ls=(0,(6,3)),zorder=5))
ax.text(43.95,38.83,"Reported 1976 failure zone\n(\u00c7ald\u0131ran\u2013Muradiye)",fontsize=8.2,
        fontweight="bold",ha="center",va="top",zorder=6,path_effects=halo)

if epi:
    ax.plot(*epi,marker="*",ms=22,mfc="#ffe14d",mec="black",mew=1.0,zorder=7,
            path_effects=[pe.withStroke(linewidth=1.2,foreground="black")])
    ax.annotate("1976 M$_w$7.3", xy=epi, xytext=(43.97,38.99), fontsize=8.6, fontweight="bold",
            ha="center", va="top", zorder=8, path_effects=halo,
            arrowprops=dict(arrowstyle="-", color="black", lw=0.9,
                            path_effects=[pe.withStroke(linewidth=2.0,foreground="white")]))

for nm,lo,la,ha in [("Van",43.38,38.49,"left"),("Erci\u015f",43.36,39.03,"right"),
        ("Ahlat",42.49,38.75,"right"),("Adilcevaz",42.74,38.80,"right"),
        ("Edremit",43.24,38.42,"right"),("Geva\u015f",43.11,38.29,"right"),
        ("\u00d6zalp",44.02,38.66,"left"),("Er\u00e7ek",43.92,38.63,"left"),
        ("Ye\u015filsu",43.62,39.02,"left"),("Mollakas\u0131m",43.52,38.73,"left"),
        ("\u00c7avu\u015ftepe",43.56,38.34,"left"),("Tatvan",42.31,38.50,"left")]:
    if W<lo<E and S<la<N:
        ax.plot(lo,la,"s",ms=4.2,mfc="white",mec="black",mew=0.8,zorder=6)
        ax.text(lo+(0.026 if ha=="left" else -0.026),la,nm,fontsize=7.3,ha=ha,va="center",
                zorder=6,path_effects=halo)

ax.plot(43.90,39.14,"s",ms=4.2,mfc="white",mec="black",mew=0.8,zorder=6)
ax.text(43.90,39.175,"\u00c7ald\u0131ran",fontsize=7.3,ha="center",va="bottom",zorder=6,path_effects=halo)
ax.plot(43.77,38.99,"s",ms=4.2,mfc="white",mec="black",mew=0.8,zorder=6)
ax.text(43.77,38.955,"Muradiye",fontsize=7.3,ha="center",va="top",zorder=6,path_effects=halo)

ax.plot(42.82,38.92,"^",ms=8,mfc="#7a5a30",mec="black",mew=0.8,zorder=6)
ax.text(42.82,38.955,"S\u00fcphan Da\u011f\u0131",fontsize=7.3,style="italic",ha="center",va="bottom",
        zorder=6,path_effects=halo)
ax.text(42.80,38.60,"Lake Van",fontsize=12,style="italic",fontweight="bold",color="#00008b",
        ha="center",va="center",zorder=6,path_effects=[pe.withStroke(linewidth=3,foreground="white")])

ax.set_xlim(W,E); ax.set_ylim(S,N); ax.set_aspect(1/COSL)
ax.set_xticks(np.arange(42.5,44.31,0.5)); ax.set_yticks(np.arange(38.0,39.41,0.5))
ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,_:f"{v:.1f}\u00b0E"))
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,_:f"{v:.1f}\u00b0N"))
ax.set_xticks(np.arange(42.3,44.31,0.1),minor=True); ax.set_yticks(np.arange(38.0,39.41,0.1),minor=True)
ax.tick_params(which="both",direction="out",length=4,labelsize=9,top=True,right=True)
ax.grid(which="major",color="0.5",lw=0.4,alpha=0.3)
for s in ax.spines.values(): s.set_linewidth(1.1)
ax.set_title("Cascade I: coseismic rockfall & landslide susceptibility \u2014 1976 \u00c7ald\u0131ran",
             fontsize=12.5,pad=10)

km=111.320*COSL; seg=40.0/km; x0,y0=E-0.16-seg,S+0.12
ax.plot([x0,x0+seg],[y0,y0],"k-",lw=3,solid_capstyle="butt",zorder=7,path_effects=[pe.withStroke(linewidth=5,foreground="white")])
for kmv in (0,10,20,30,40):
    xt=x0+seg*kmv/40.0
    ax.plot([xt,xt],[y0,y0+0.032],"k-",lw=1.8,zorder=7,path_effects=[pe.withStroke(linewidth=3.4,foreground="white")])
    ax.text(xt,y0+0.05,f"{kmv}",ha="center",va="bottom",fontsize=7,zorder=7,path_effects=halo)
ax.text(x0+seg+0.03,y0,"km",ha="left",va="center",fontsize=7.5,zorder=7,path_effects=halo)
ax.annotate("N",xy=(W+0.13,N-0.08),xytext=(W+0.13,N-0.26),ha="center",va="center",fontsize=12,
            fontweight="bold",zorder=7,path_effects=halo,arrowprops=dict(arrowstyle="-|>",lw=2.2,color="black"))

leg=ax.legend(handles=[
    Line2D([0],[0],color="#5a1e6e",lw=2.6,label="\u00c7ald\u0131ran Fault"),
    Line2D([0],[0],color="#111",lw=1.8,ls=(0,(6,3)),label="Reported failure zone"),
    Line2D([0],[0],marker="*",linestyle="none",markersize=13,markerfacecolor="#ffe14d",
           markeredgecolor="black",label="1976 epicentre (IEB)")],
    loc="lower left",fontsize=8.3,framealpha=0.92,borderpad=0.7)
leg.get_frame().set_edgecolor("0.4")
cax=ax.inset_axes([1.03,0.0,0.028,1.0]); cb=fig.colorbar(im,cax=cax)
cb.set_label("Coseismic slope-failure susceptibility",fontsize=10)
cb.ax.yaxis.set_major_locator(MultipleLocator(1.0))
cb.ax.yaxis.set_minor_locator(MultipleLocator(0.1))
cb.ax.tick_params(which="minor",length=2.5)

ax.text(0.0,-0.075,
    "Susceptibility: from GEBCO 2026 terrain (slope, relief) and \u00c7ald\u0131ran-fault proximity, "
    "based on the trained ML model.\n"
    "Relief: GEBCO grey hillshade. Epicentre: IEB. Reported failure zone: representative "
    "(\u00c7ald\u0131ran\u2013Muradiye, after Toks\u00f6z et al. 1977). Plotted with Matplotlib. Source: authors.",
    transform=ax.transAxes,fontsize=6.0,color="0.3",ha="left",va="top",linespacing=1.35)

fig.canvas.draw(); rend=fig.canvas.get_renderer()
tx=[t for t in ax.texts if t.get_text().strip()]
bx=[(t.get_text().replace("\n"," ")[:16], t.get_window_extent(rend)) for t in tx]
ov=[]
for i in range(len(bx)):
    for j in range(i+1,len(bx)):
        a,b=bx[i][1],bx[j][1]
        if not(a.x1<=b.x0 or a.x0>=b.x1 or a.y1<=b.y0 or a.y0>=b.y1): ov.append((bx[i][0],bx[j][0]))
print("overlapping label pairs:", ov if ov else "none")

fig.savefig("outputs/fig10_cascade1_rockfall_landslide.pdf",bbox_inches="tight")
fig.savefig("outputs/fig10_cascade1_rockfall_landslide.png",dpi=600,bbox_inches="tight")
print("wrote fig10_cascade1_rockfall_landslide.pdf/.png")
