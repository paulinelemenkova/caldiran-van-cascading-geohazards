import csv
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable as SM
from matplotlib.patches import Polygon, Patch
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import rasterio, shapefile
from pyproj import CRS, Transformer
from obspy.imaging.beachball import beach

DEM = "data/gebco_2026_n39_4_s38_0_w42_3_e44_3_geotiff.tif"
EQ  = "data/IEB_Turkey_5000_events.csv"
GEO = "geodata/geo4_2l/geo4_2l"

ds = rasterio.open(DEM); z = ds.read(1).astype(float)
if ds.nodata is not None: z[z == ds.nodata] = np.nan
b = ds.bounds; W,E,S,N = b.left,b.right,b.bottom,b.top; MID = 0.5*(S+N)
COSL = np.cos(np.deg2rad(MID))
dx_m = ds.res[0]*111320*COSL; dy_m = ds.res[1]*111320
hs = LightSource(azdeg=315, altdeg=45).hillshade(z, vert_exag=1.5, dx=dx_m, dy=dy_m)
grey = LinearSegmentedColormap.from_list("greylite", ["#8f8f8f", "#ffffff"])

src = CRS.from_wkt(open(GEO+".prj").read()); to_ll = Transformer.from_crs(src,4326,always_xy=True)
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

lon,lat,dep,mag=[],[],[],[]
for row in csv.DictReader(open(EQ,newline="")):
    try: lo=float(row["Lon"]);la=float(row["Lat"]);d=float(row["Depth"]);m=float(row["Mag"])
    except: continue
    if W<=lo<=E and S<=la<=N: lon.append(lo);lat.append(la);dep.append(d);mag.append(m)
lon=np.array(lon);lat=np.array(lat);dep=np.array(dep);mag=np.array(mag)
o=np.argsort(mag); lon,lat,dep,mag=lon[o],lat[o],dep[o],mag[o]
def msize(m): return 9.0*(2.4**(m-4.0))
print(f"events:{len(lon)} depth {dep.min():.0f}-{dep.max():.0f} mag {mag.min():.1f}-{mag.max():.1f}")

REV="#1f3f8f"; SS="#8b1a1a"
principal=[
 (44.029,39.121,1976,7.3,"\u00c7ald\u0131ran",[245,82,-176],SS,(43.72,39.31)),
 (43.508,38.721,2011,7.1,"Van",       [246,38, 71],REV,(43.99,38.96)),
 (43.229,38.429,2011,5.6,"Edremit",   [ 98,46, 74],REV,(43.62,38.25)),
]

mpl.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42})
fig, ax = plt.subplots(figsize=(11.8, 8.7))
ax.imshow(hs, extent=[W,E,S,N], origin="upper", cmap=grey, vmin=0, vmax=1,
          zorder=0, aspect="auto", interpolation="bilinear")
if lake:
    ax.add_collection(PatchCollection([Polygon(a,closed=True) for a in lake],
        facecolor="#cfe3f0", edgecolor="#2f6d8f", linewidths=0.7, alpha=0.7, zorder=1))

halo=[pe.withStroke(linewidth=2.4,foreground="white")]

dnorm=Normalize(0,50)
sc=ax.scatter(lon,lat,c=dep,s=msize(mag),cmap="plasma",norm=dnorm,edgecolors="black",
              linewidths=0.35,alpha=0.85,zorder=4)

wx=0.12
for elo,ela,yr,mw,place,fm,col,(tx,ty) in principal:
    w=(wx*(0.85+0.10*(mw-6)), wx*(0.85+0.10*(mw-6))*COSL)
    bb=beach(fm, xy=(elo,ela), width=w, facecolor=col, edgecolor="black",
             linewidth=0.8, zorder=6)
    ax.add_collection(bb)
    ax.annotate(f"{yr}  M$_w${mw}  {place}", xy=(elo,ela), xytext=(tx,ty),
                fontsize=8.6, fontweight="bold", color="#111", ha="center", va="center",
                zorder=8, path_effects=[pe.withStroke(linewidth=2.8,foreground="white")],
                arrowprops=dict(arrowstyle="-", color="#111", lw=1.0,
                                path_effects=[pe.withStroke(linewidth=2.4,foreground="white")]))

for nm,lo,la,ha in [("Van",43.38,38.49,"left"),("Erci\u015f",43.36,39.03,"right"),
        ("\u00c7ald\u0131ran",43.90,39.14,"left"),("Muradiye",43.77,38.99,"left"),
        ("Edremit",43.24,38.42,"right"),("Ahlat",42.49,38.75,"right"),
        ("Adilcevaz",42.74,38.80,"right"),("\u00d6zalp",44.02,38.66,"left"),
        ("Ba\u015fkale",44.01,38.04,"left"),("Patnos",42.86,39.23,"right"),
        ("Malazgirt",42.54,39.14,"right"),("Geva\u015f",43.11,38.29,"right")]:
    if W<lo<E and S<la<N:
        ax.plot(lo,la,"s",ms=4.2,mfc="white",mec="black",mew=0.8,zorder=5)
        ax.text(lo+(0.028 if ha=="left" else -0.028),la,nm,fontsize=7.3,ha=ha,va="center",
                zorder=5,path_effects=halo)
ax.text(42.80,38.60,"Lake Van",fontsize=13,style="italic",fontweight="bold",color="#00008b",
        ha="center",va="center",zorder=6,path_effects=[pe.withStroke(linewidth=3,foreground="white")])

ax.set_xlim(W,E); ax.set_ylim(S,N); ax.set_aspect(1/COSL)
ax.set_xticks(np.arange(42.5,44.31,0.5)); ax.set_yticks(np.arange(38.0,39.41,0.5))
ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,_:f"{v:.1f}\u00b0E"))
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,_:f"{v:.1f}\u00b0N"))
ax.set_xticks(np.arange(42.3,44.31,0.1),minor=True); ax.set_yticks(np.arange(38.0,39.41,0.1),minor=True)
ax.tick_params(which="both",direction="out",length=4,labelsize=9,top=True,right=True)
ax.grid(True,which="major",color="0.5",lw=0.4,alpha=0.3)
for s in ax.spines.values(): s.set_linewidth(1.1)
ax.set_title("Historical and instrumental seismicity of the "
             "\u00c7ald\u0131ran\u2013Erci\u015f\u2013Van region", fontsize=12.5, pad=10)

km=111.320*COSL; seg=40.0/km; x0,y0=E-0.14-seg,S+0.10
ax.plot([x0,x0+seg],[y0,y0],"k-",lw=3,solid_capstyle="butt",zorder=7,
        path_effects=[pe.withStroke(linewidth=5,foreground="white")])
ax.text(x0+seg/2,y0+0.035,"40 km",ha="center",va="bottom",fontsize=8.5,zorder=7,path_effects=halo)
ax.annotate("N",xy=(E-0.12,N-0.08),xytext=(E-0.12,N-0.26),ha="center",va="center",fontsize=12,
            fontweight="bold",zorder=7,path_effects=halo,arrowprops=dict(arrowstyle="-|>",lw=2.2,color="black"))

mh=[Line2D([0],[0],marker="o",linestyle="none",markersize=np.sqrt(msize(m)),
    markerfacecolor="0.7",markeredgecolor="black",markeredgewidth=0.5,label=f"M {m}") for m in (4,5,6,7)]
mh += [Patch(facecolor=REV,edgecolor="black",label="Reverse mechanism"),
       Patch(facecolor=SS, edgecolor="black",label="Strike-slip mechanism")]
leg=ax.legend(handles=mh,loc="lower left",fontsize=8.3,framealpha=0.92,borderpad=0.8,
              labelspacing=0.9,title="Magnitude / focal mechanism",title_fontsize=8.8)
leg.get_frame().set_edgecolor("0.4")

cax=ax.inset_axes([1.03,0.0,0.028,1.0]); cb=fig.colorbar(sc,cax=cax,extend="max")
cb.set_label("Focal depth (km)",fontsize=10); cb.ax.invert_yaxis()

ax.text(0.0,-0.075,
    "Relief: GEBCO 2026 (grey hillshade). Seismicity: IRIS Earthquake Browser (IEB), 1970\u20132026.\n"
    "Focal mechanisms of the three principal events after published solutions (GCMT; Kalafat et al., 2014).\n"
    "Plotted with Matplotlib + ObsPy. Source: authors.",
    transform=ax.transAxes,fontsize=6.8,color="0.3",ha="left",va="top",linespacing=1.35)

fig.savefig("outputs/fig06_seismicity_map.pdf",bbox_inches="tight")
fig.savefig("outputs/fig06_seismicity_map.png",dpi=600,bbox_inches="tight")
print("wrote fig06_seismicity_map.pdf/.png")
