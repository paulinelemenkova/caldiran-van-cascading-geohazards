import csv
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrow
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
import rasterio, shapefile
from pyproj import CRS, Transformer

DEM = "data/gebco_2026_n39_4_s38_0_w42_3_e44_3_geotiff.tif"
EQ  = "data/IEB_Turkey_5000_events.csv"
GEO = "geodata/geo4_2l/geo4_2l"

A = (43.60, 38.48)
B = (43.34, 38.92)
SWATH = 15.0

ds = rasterio.open(DEM); z = ds.read(1).astype(float)
if ds.nodata is not None: z[z == ds.nodata] = np.nan
Wd,Ed,Sd,Nd = ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top
MID = 38.7; kx = 111.32*np.cos(np.deg2rad(MID)); ky = 111.0

def to_km(lon, lat): return ((lon-A[0])*kx, (lat-A[1])*ky)
Bx, By = to_km(*B); L = np.hypot(Bx, By)
u = np.array([Bx/L, By/L]); nvec = np.array([-u[1], u[0]])

def sample_dem(lon, lat):
    c = int(round((lon-Wd)/ds.res[0])); r = int(round((Nd-lat)/ds.res[1]))
    c = min(max(c,0), z.shape[1]-1); r = min(max(r,0), z.shape[0]-1)
    return z[r, c]

tt = np.linspace(0, 1, 500)
plon = A[0] + tt*(B[0]-A[0]); plat = A[1] + tt*(B[1]-A[1])
palong = tt*L
pelev = np.array([sample_dem(lo, la) for lo, la in zip(plon, plat)])/1000.0

src = CRS.from_wkt(open(GEO+".prj").read()); to_ll = Transformer.from_crs(src,4326,always_xy=True)
lakepaths = []
r = shapefile.Reader(GEO+".shp"); gi=[f[0] for f in r.fields[1:]].index("GLG")
for sr in r.iterShapeRecords():
    if sr.record[gi] != "H2O": continue
    sh=sr.shape; parts=list(sh.parts)+[len(sh.points)]
    for i in range(len(parts)-1):
        seg=sh.points[parts[i]:parts[i+1]]
        if len(seg)>=3:
            xs=[p[0] for p in seg]; ys=[p[1] for p in seg]
            lo,la=to_ll.transform(xs,ys); lakepaths.append(MplPath(np.column_stack([lo,la])))
pts = np.column_stack([plon, plat])
lakemask = np.zeros(len(pts), bool)
for pth in lakepaths: lakemask |= pth.contains_points(pts)

ealong, edepth, emag = [], [], []
for row in csv.DictReader(open(EQ, newline="")):
    try: lo=float(row["Lon"]); la=float(row["Lat"]); d=float(row["Depth"]); m=float(row["Mag"])
    except: continue
    pk = np.array(to_km(lo, la)); al = pk@u; pe_ = pk@nvec
    if 0 <= al <= L and abs(pe_) <= SWATH:
        ealong.append(al); edepth.append(d); emag.append(m)
ealong=np.array(ealong); edepth=np.array(edepth); emag=np.array(emag)
print(f"profile L={L:.1f} km | topo {pelev.min():.2f}-{pelev.max():.2f} km | "
      f"projected events={len(ealong)} | lake samples={int(lakemask.sum())}")

Ftip = np.array([22.0, -4.0]); Fbot = np.array([42.0, -28.0])
def on_fault(depth):
    s = (depth - Ftip[1]) / (Fbot[1] - Ftip[1]); return Ftip + s*(Fbot - Ftip)
slip_top = on_fault(-10.0); slip_bot = on_fault(-25.0)
hypo = on_fault(-18.0)

mpl.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42})
fig, ax = plt.subplots(figsize=(12.5, 8.4))
ax.set_aspect("equal")
ZTOP, ZBOT = 4.0, -46.0
halo = [pe.withStroke(linewidth=2.4, foreground="white")]

crust = np.vstack([np.column_stack([palong, pelev]),
                   [[L, ZBOT], [0, ZBOT]]])
ax.add_patch(Polygon(crust, closed=True, facecolor="#e9e2d6", edgecolor="none", zorder=0))
ax.plot([0, L], [-44, -44], ls=(0,(7,4)), color="#555", lw=1.3, zorder=1)
ax.text(L*0.5, -44, "Moho (~45 km depth)", fontsize=9, color="#444", ha="center",
        va="bottom", zorder=2, path_effects=halo)
ax.text(2, ZBOT+9, "Upper crust", fontsize=10, color="#7a6f57", ha="left", style="italic", zorder=2)

b0 = 30.0
top_at = lambda a: np.interp(a, palong, pelev)
basin = [[b0, top_at(b0)-0.2], [L, top_at(L)], [L, -3.2], [b0, -0.3]]
ax.add_patch(Polygon(basin, closed=True, facecolor="#f6e6a8", edgecolor="#b8a24a",
                     lw=0.8, alpha=0.9, zorder=1))
ax.text((b0+L)/2, -1.7, "Neogene\u2013Quaternary\nbasin fill", fontsize=8.5, ha="center",
        va="center", color="#6b5d13", zorder=2, path_effects=halo)

ax.plot([Ftip[0], Fbot[0]], [Ftip[1], Fbot[1]], color="#7b3294", lw=3.2, zorder=3,
        solid_capstyle="round", path_effects=[pe.withStroke(linewidth=5, foreground="white")])
ax.plot([Ftip[0], Ftip[0]-4], [Ftip[1], top_at(Ftip[0]-4)], color="#7b3294", lw=1.4,
        ls=(0,(3,3)), zorder=3)
ax.text(Ftip[0]-4, top_at(Ftip[0]-4)+0.4, "blind tip\n(no surface rupture)", fontsize=8,
        color="#5a1e6e", ha="center", va="bottom", zorder=6, path_effects=halo)

ax.plot([slip_top[0], slip_bot[0]], [slip_top[1], slip_bot[1]], color="#d7191c", lw=7,
        solid_capstyle="round", zorder=3.2, alpha=0.85)
ax.text(slip_bot[0]+1.5, -22, "2011 coseismic\nslip patch (10\u201325 km)", fontsize=8.5,
        color="#a01015", ha="left", va="center", zorder=6, path_effects=halo)

mid = on_fault(-15.0); perp = np.array([np.cos(np.deg2rad(50))*0, 0])
duv = (Fbot-Ftip)/np.linalg.norm(Fbot-Ftip)
nrm = np.array([-duv[1], duv[0]])
ax.annotate("", xy=mid-duv*4+nrm*1.2, xytext=mid+duv*2+nrm*1.2,
            arrowprops=dict(arrowstyle="-|>", color="#222", lw=2.0), zorder=5)
ax.annotate("", xy=mid+duv*4-nrm*1.2, xytext=mid-duv*2-nrm*1.2,
            arrowprops=dict(arrowstyle="-|>", color="#222", lw=2.0), zorder=5)

sz = 8*(2.2**(emag-4.0))
ax.scatter(ealong, -edepth, s=sz, facecolor="#3b4cc0", edgecolor="black", linewidths=0.4,
           alpha=0.8, zorder=4)

ax.plot(hypo[0], hypo[1], marker="*", ms=22, mfc="#ffd21e", mec="black", mew=1.0, zorder=7,
        path_effects=[pe.withStroke(linewidth=1.2, foreground="black")])
ax.text(hypo[0]-1.2, hypo[1], "Mainshock\nM$_w$7.1 (18 km)", fontsize=8.5, ha="right",
        va="center", fontweight="bold", zorder=7, path_effects=halo)

ax.plot(palong, pelev, color="#3a2f1a", lw=1.6, zorder=2)
if lakemask.any():
    la_elev = np.where(lakemask, np.maximum(pelev, 1.60), np.nan)
    ax.fill_between(palong, la_elev, 1.55, where=lakemask, color="#7fb2d6", alpha=0.8, zorder=2)
    ci = palong[lakemask]
    ax.text(ci.mean(), 2.5, "Lake Van", fontsize=11, style="italic", fontweight="bold",
            color="#00008b", ha="center", zorder=6, path_effects=halo)

ax.text(10, -6, "Hanging wall (SE)\n\u2191 uplift", fontsize=9, color="#333", ha="center",
        va="center", fontweight="bold", zorder=6, path_effects=halo)
ax.annotate("Van Fault Zone\n(blind reverse)", xy=on_fault(-7.5), xytext=(10.5,-14),
            fontsize=9.5, fontweight="bold", color="#5a1e6e", ha="center", va="center",
            zorder=6, path_effects=halo,
            arrowprops=dict(arrowstyle="->", color="#5a1e6e", lw=1.4))
ax.text(L-6, -8, "Footwall (NW)", fontsize=9, color="#333", ha="center", va="center",
        fontweight="bold", zorder=6, path_effects=halo)

ax.set_xlim(0, L); ax.set_ylim(ZBOT, ZTOP)
ax.set_xlabel("Distance along profile (km)", fontsize=11)
ax.set_ylabel("Elevation / depth (km)", fontsize=11)
ax.set_yticks(np.arange(0, ZBOT-1, -10))
ax.tick_params(labelsize=9)
ax.axhline(0, color="0.6", lw=0.6, ls=":", zorder=0)
ax.text(0.6, ZTOP-0.4, "SE  (A)", fontsize=11, fontweight="bold", ha="left", va="top", path_effects=halo)
ax.text(L-0.6, ZTOP-0.4, "(A\u2032)  NW", fontsize=11, fontweight="bold", ha="right", va="top", path_effects=halo)
ax.set_title("Interpretive cross-section across the Van Fault Zone", fontsize=13, pad=8)
for s in ax.spines.values(): s.set_linewidth(1.1)

mags=[4,5,6]
mh=[Line2D([0],[0],marker="o",linestyle="none",markersize=np.sqrt(8*(2.2**(m-4.0))),
    markerfacecolor="#3b4cc0",markeredgecolor="black",markeredgewidth=0.5) for m in mags]
mh.append(Line2D([0],[0],marker="*",linestyle="none",markersize=15,markerfacecolor="#ffd21e",
    markeredgecolor="black",label="Mainshock"))
leg=ax.legend(mh,[f"M {m}" for m in mags]+["Mainshock"], title=f"Projected seismicity\n(IEB, \u00b1{SWATH:.0f} km swath)",
    loc="lower right", fontsize=8.5, title_fontsize=8.5, framealpha=0.92, labelspacing=1.0,
    borderpad=0.8); leg.get_frame().set_edgecolor("0.4")

ax.text(0.0, -0.075,
    f"Topography: GEBCO 2026 along {A[0]:.2f}\u00b0E,{A[1]:.2f}\u00b0N \u2192 {B[0]:.2f}\u00b0E,{B[1]:.2f}\u00b0N. "
    "Seismicity projected from IEB.\n"
    "Fault geometry, slip patch and basin fill interpretive, after Mackenzie et al. (2016).\n"
    "Vertical scale = horizontal scale (no exaggeration). Plotted with Matplotlib. Source: authors.",
    transform=ax.transAxes, fontsize=7.2, color="0.3", ha="left", va="top", linespacing=1.35)

fig.savefig("outputs/fig05_cross_section.pdf", bbox_inches="tight")
fig.savefig("outputs/fig05_cross_section.png", dpi=600, bbox_inches="tight")
print("wrote fig05_cross_section.pdf/.png")
