import csv
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable as SM
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import rasterio, shapefile
from pyproj import CRS, Transformer

DEM = "data/gebco_2026_n39_4_s38_0_w42_3_e44_3_geotiff.tif"
EQ  = "data/IEB_Turkey_5000_events.csv"
GEO = "geodata/geo4_2l/geo4_2l"
FLT = "geodata/flt4_2l/flt4_2l"

ds = rasterio.open(DEM); z = ds.read(1).astype(float)
if ds.nodata is not None: z[z == ds.nodata] = np.nan
b = ds.bounds; W,E,S,N = b.left,b.right,b.bottom,b.top; MID = 0.5*(S+N)
zmin,zmax = np.nanmin(z), np.nanmax(z)
dx_m = ds.res[0]*111320*np.cos(np.deg2rad(MID)); dy_m = ds.res[1]*111320

_dem1 = [(0.000,(51,102,0)),(0.125,(129,195,31)),(0.250,(255,255,204)),
         (0.500,(244,189,69)),(0.625,(102,51,12)),(0.750,(102,51,0)),(1.000,(255,254,253))]
terr = LinearSegmentedColormap.from_list("dem1",
        [(p,(r/255,g/255,b/255)) for p,(r,g,b) in _dem1])
rgb = LightSource(azdeg=315,altdeg=45).shade(z, cmap=terr, blend_mode="soft",
        vert_exag=1.6, dx=dx_m, dy=dy_m, vmin=zmin, vmax=zmax)

src = CRS.from_wkt(open(GEO+".prj").read()); to_ll = Transformer.from_crs(src,4326,always_xy=True)

def read_lines(base, want_type=False):
    r = shapefile.Reader(base+".shp"); flds=[f[0] for f in r.fields[1:]]
    ti = flds.index("TYPE") if "TYPE" in flds else None
    out=[]
    for sr in r.iterShapeRecords():
        sh=sr.shape; parts=list(sh.parts)+[len(sh.points)]
        for i in range(len(parts)-1):
            seg=sh.points[parts[i]:parts[i+1]]
            if len(seg)<2: continue
            xs=[p[0] for p in seg]; ys=[p[1] for p in seg]
            lo,la=to_ll.transform(xs,ys); a=np.column_stack([lo,la])
            if a[:,0].max()<W or a[:,0].min()>E or a[:,1].max()<S or a[:,1].min()>N: continue
            t=(sr.record[ti] or "flt").strip() if ti is not None else "flt"
            out.append((t,a))
    return out

mapped = read_lines(FLT)
lake=[]
r=shapefile.Reader(GEO+".shp"); gi=[f[0] for f in r.fields[1:]].index("GLG")
for sr in r.iterShapeRecords():
    if sr.record[gi]!="H2O": continue
    sh=sr.shape; parts=list(sh.parts)+[len(sh.points)]
    for i in range(len(parts)-1):
        seg=sh.points[parts[i]:parts[i+1]]
        if len(seg)>=3:
            xs=[p[0] for p in seg]; ys=[p[1] for p in seg]
            lo,la=to_ll.transform(xs,ys); lake.append(np.column_stack([lo,la]))

eqs=[]
for row in csv.DictReader(open(EQ,newline="")):
    try: lo=float(row["Lon"]);la=float(row["Lat"]);m=float(row["Mag"]);y=int(row["Year"])
    except: continue
    if W<=lo<=E and S<=la<=N and m>=5.5: eqs.append((lo,la,m,y))

CALD = np.array([(43.72,39.00),(43.90,39.09),(44.09,39.18),(44.30,39.27)])
ERC  = np.array([(43.06,38.96),(43.24,39.03),(43.44,39.12)])
VAN  = np.array([(43.16,38.55),(43.40,38.55),(43.66,38.58)])

mpl.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42})
fig,ax=plt.subplots(figsize=(11.6,8.6))
ax.imshow(rgb, extent=[W,E,S,N], origin="upper", zorder=0, aspect="auto", interpolation="bilinear")

_xs = W + (np.arange(z.shape[1])+0.5)*ds.res[0]
_ys = N - (np.arange(z.shape[0])+0.5)*ds.res[1]
ax.contour(_xs, _ys[::-1], np.flipud(z), levels=np.arange(1000,4001,500),
           colors="#7a4a1e", linewidths=0.35, alpha=0.6, zorder=1)
if lake:
    ax.add_collection(PatchCollection([Polygon(a,closed=True) for a in lake],
        facecolor="#a9cfe7", edgecolor="#2f6d8f", linewidths=0.7, alpha=0.5, zorder=1.5))

halo=[pe.withStroke(linewidth=2.4,foreground="white")]
ASP=1/np.cos(np.deg2rad(MID))

def fline(pts,color,lw,z):
    ax.plot(pts[:,0],pts[:,1],"-",color=color,lw=lw,solid_capstyle="round",zorder=z,
            path_effects=[pe.withStroke(linewidth=lw+1.8,foreground="white")])

def _mid_dir(P):
    seg=np.hypot(np.diff(P[:,0]),np.diff(P[:,1])*ASP); cum=np.concatenate([[0],np.cumsum(seg)])
    h=cum[-1]/2; i=min(max(np.searchsorted(cum,h)-1,0),len(P)-2)
    p0,p1=P[i],P[i+1]; u=np.array([(p1-p0)[0],(p1-p0)[1]*ASP]); u=u/np.hypot(*u)
    mid=(p0+p1)/2; return mid,u

def strikeslip(P,color,dextral,z=6):
    mid,u=_mid_dir(P); n=np.array([-u[1],u[0]])
    off,L=0.05,0.13; s=1 if dextral else -1
    for side in (+1,-1):
        base=np.array([mid[0]+n[0]*off*side, mid[1]+n[1]*off*side/ASP])
        head=np.array([base[0]+u[0]*L*side*s, base[1]+u[1]*L*side*s/ASP])
        ax.annotate("",xy=head,xytext=base,zorder=z,
            arrowprops=dict(arrowstyle="-|>",color=color,lw=1.8,
                            path_effects=[pe.withStroke(linewidth=3,foreground="white")]))

def teeth(P,color,side=+1,z=6,size=0.028,spacing=0.075):
    seg=np.hypot(np.diff(P[:,0]),np.diff(P[:,1])*ASP); cum=np.concatenate([[0],np.cumsum(seg)])
    d=spacing*0.6
    while d<cum[-1]:
        i=min(max(np.searchsorted(cum,d)-1,0),len(P)-2)
        t=(d-cum[i])/(seg[i] if seg[i]>0 else 1); base=P[i]+(P[i+1]-P[i])*t
        u=np.array([(P[i+1]-P[i])[0],(P[i+1]-P[i])[1]*ASP]); u=u/np.hypot(*u); n=np.array([-u[1],u[0]])*side
        apex=np.array([base[0]+n[0]*size, base[1]+n[1]*size/ASP])
        b1=np.array([base[0]-u[0]*size*0.6, base[1]-u[1]*size*0.6/ASP])
        b2=np.array([base[0]+u[0]*size*0.6, base[1]+u[1]*size*0.6/ASP])
        ax.add_patch(Polygon([b1,b2,apex],closed=True,facecolor=color,edgecolor=color,lw=0.3,zorder=z))
        d+=spacing

ax.add_collection(LineCollection([a for _,a in mapped], colors="#222222", linewidths=1.0,
                  zorder=3, path_effects=[pe.withStroke(linewidth=2.0,foreground="white")]))

C_CALD,C_ERC,C_VAN = "#d7191c","#e08214","#7b3294"
fline(CALD,C_CALD,2.6,4); strikeslip(CALD,C_CALD,dextral=True)
fline(ERC, C_ERC, 2.6,4); strikeslip(ERC, C_ERC, dextral=False)
fline(VAN, C_VAN, 2.6,4); teeth(VAN, C_VAN, side=+1)
ax.text(43.84,39.033,"\u00c7ald\u0131ran F. (dextral)",color=C_CALD,fontsize=9,fontweight="bold",
        ha="center",va="top",rotation=27,zorder=7,path_effects=halo)
ax.text(43.29,39.105,"Erci\u015f F. (sinistral)",color=C_ERC,fontsize=9,fontweight="bold",
        ha="center",va="bottom",rotation=24,zorder=7,path_effects=halo)
ax.text(43.40,38.50,"Van Fault Zone (reverse)",color=C_VAN,fontsize=9,fontweight="bold",
        ha="center",va="top",zorder=7,path_effects=halo)

for lo,la,m,y in sorted(eqs,key=lambda e:e[2]):
    ax.plot(lo,la,marker="*",ms=6+3.0*(m-5.0),mfc="#ffe14d",mec="black",mew=0.8,zorder=6,
            path_effects=[pe.withStroke(linewidth=1.0,foreground="black")])
for lo,la,m,y in eqs:
    if m>=7.0:
        ax.text(lo,la+0.05,f"{y}  M{m}",fontsize=8.5,fontweight="bold",ha="center",va="bottom",
                zorder=7,path_effects=halo)

for nm,lo,la,ha in [("Van",43.38,38.49,"left"),("Erci\u015f",43.36,39.03,"right"),
        ("\u00c7ald\u0131ran",43.90,39.14,"left"),("Muradiye",43.77,38.99,"left"),
        ("Edremit",43.24,38.42,"right"),("Ahlat",42.49,38.75,"right"),
        ("Adilcevaz",42.74,38.80,"right"),("Geva\u015f",43.10,38.29,"right"),
        ("G\u00fcrp\u0131nar",43.39,38.31,"left"),("\u00d6zalp",44.02,38.66,"left"),
        ("Ba\u015fkale",44.01,38.04,"left")]:
    if W<lo<E and S<la<N:
        ax.plot(lo,la,"s",ms=5,mfc="white",mec="black",mew=0.9,zorder=6)
        ax.text(lo+(0.03 if ha=="left" else -0.03),la,nm,fontsize=8.3,ha=ha,va="center",
                zorder=6,path_effects=halo)
ax.text(42.80,38.60,"Lake Van",fontsize=14,style="italic",fontweight="bold",color="#00008b",
        ha="center",va="center",zorder=6,path_effects=[pe.withStroke(linewidth=3,foreground="white")])

ax.set_xlim(W,E); ax.set_ylim(S,N); ax.set_aspect(ASP)
ax.set_xticks(np.arange(42.5,44.31,0.5)); ax.set_yticks(np.arange(38.0,39.41,0.5))
ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,_:f"{v:.1f}\u00b0E"))
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,_:f"{v:.1f}\u00b0N"))
ax.set_xticks(np.arange(42.3,44.31,0.1),minor=True); ax.set_yticks(np.arange(38.0,39.41,0.1),minor=True)
ax.tick_params(which="both",direction="out",length=4,labelsize=9,top=True,right=True)
ax.grid(True,which="major",color="0.5",lw=0.4,alpha=0.35)
for s in ax.spines.values(): s.set_linewidth(1.1)
ax.set_title("Active faults and principal earthquakes of the "
             "\u00c7ald\u0131ran\u2013Erci\u015f\u2013Van region",fontsize=12.5,pad=10)

km=111.320*np.cos(np.deg2rad(MID)); seg=40.0/km; x0,y0=E-0.14-seg,S+0.10
ax.plot([x0,x0+seg],[y0,y0],"k-",lw=3,solid_capstyle="butt",zorder=7,
        path_effects=[pe.withStroke(linewidth=5,foreground="white")])
ax.text(x0+seg/2,y0+0.035,"40 km",ha="center",va="bottom",fontsize=8.5,zorder=7,path_effects=halo)
ax.annotate("N",xy=(E-0.12,N-0.08),xytext=(E-0.12,N-0.26),ha="center",va="center",fontsize=12,
            fontweight="bold",zorder=7,path_effects=halo,arrowprops=dict(arrowstyle="-|>",lw=2.2,color="black"))

handles=[
 Line2D([0],[0],color="#222222",lw=1.4,label="Mapped fault (regional data)"),
 Line2D([0],[0],color=C_CALD,lw=2.6,label="\u00c7ald\u0131ran F. \u2014 dextral (schematic)"),
 Line2D([0],[0],color=C_ERC,lw=2.6,label="Erci\u015f F. \u2014 sinistral (schematic)"),
 Line2D([0],[0],color=C_VAN,lw=2.6,label="Van Fault Zone \u2014 reverse (schematic)"),
 Line2D([0],[0],marker="*",linestyle="none",markersize=12,markerfacecolor="#ffe14d",
        markeredgecolor="black",label="Earthquake M\u22655.5 (IEB)")]
leg=ax.legend(handles=handles,loc="lower left",fontsize=8.2,framealpha=0.92,borderpad=0.8,
              handlelength=2.0); leg.get_frame().set_edgecolor("0.4")

cax=ax.inset_axes([0.0,-0.10,1.0,0.03]); sm=SM(norm=Normalize(zmin,zmax),cmap=terr); sm.set_array([])
cb=fig.colorbar(sm,cax=cax,orientation="horizontal"); cb.set_label("Elevation (m)",fontsize=10)

ax.text(0.0,-0.19,
 "Relief: GEBCO 2026. Thin faults: provided regional (USGS) layer. Bold traces: generalized active "
 "faults (schematic) after Emre et al. (2018) and Ko\u00e7yi\u011fit (2013).\n"
 "Epicentres: IEB. Plotted with Matplotlib. Source: authors.",
 transform=ax.transAxes,fontsize=6.4,color="0.3",ha="left",va="top")

fig.savefig("outputs/fig04_active_faults.pdf",bbox_inches="tight")
fig.savefig("outputs/fig04_active_faults.png",dpi=600,bbox_inches="tight")
print(f"wrote fig04 | mapped faults:{len(mapped)} | EQ stars:{len(eqs)} | lake rings:{len(lake)}")
