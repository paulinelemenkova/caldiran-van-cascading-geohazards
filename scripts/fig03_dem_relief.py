import csv
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable as SM
import matplotlib.patheffects as pe
import rasterio
import shapefile
from pyproj import CRS, Transformer

DEM = "data/gebco_2026_n39_4_s38_0_w42_3_e44_3_geotiff.tif"
EQ  = "data/IEB_Turkey_5000_events.csv"
GEO = "geodata/geo4_2l/geo4_2l"

ds = rasterio.open(DEM)
z = ds.read(1).astype(float)
if ds.nodata is not None:
    z[z == ds.nodata] = np.nan
b = ds.bounds; W, E, S, N = b.left, b.right, b.bottom, b.top
MIDLAT = 0.5 * (S + N)
zmin, zmax = np.nanmin(z), np.nanmax(z)
dy_m = ds.res[1] * 111320.0
dx_m = ds.res[0] * 111320.0 * np.cos(np.deg2rad(MIDLAT))

_t = plt.cm.terrain(np.linspace(0.25, 1.0, 256))
terr = LinearSegmentedColormap.from_list("terr", _t)
ls = LightSource(azdeg=315, altdeg=45)
rgb = ls.shade(z, cmap=terr, blend_mode="soft", vert_exag=1.6,
               dx=dx_m, dy=dy_m, vmin=zmin, vmax=zmax)

lake = []
try:
    src = CRS.from_wkt(open(GEO + ".prj").read())
    to_ll = Transformer.from_crs(src, 4326, always_xy=True)
    r = shapefile.Reader(GEO + ".shp"); gi = [f[0] for f in r.fields[1:]].index("GLG")
    for sr in r.iterShapeRecords():
        if sr.record[gi] != "H2O":
            continue
        sh = sr.shape; parts = list(sh.parts) + [len(sh.points)]
        for i in range(len(parts) - 1):
            seg = sh.points[parts[i]:parts[i+1]]
            if len(seg) >= 3:
                xs = [p[0] for p in seg]; ys = [p[1] for p in seg]
                lo, la = to_ll.transform(xs, ys); lake.append(np.column_stack([lo, la]))
except Exception as e:
    print("lake outline skipped:", e)

lon, lat, dep, mag = [], [], [], []
with open(EQ, newline="") as f:
    for row in csv.DictReader(f):
        try:
            lo = float(row["Lon"]); la = float(row["Lat"])
            d = float(row["Depth"]); m = float(row["Mag"])
        except (ValueError, KeyError):
            continue
        if W <= lo <= E and S <= la <= N:
            lon.append(lo); lat.append(la); dep.append(d); mag.append(m)
lon = np.array(lon); lat = np.array(lat); dep = np.array(dep); mag = np.array(mag)
order = np.argsort(mag)
lon, lat, dep, mag = lon[order], lat[order], dep[order], mag[order]
def msize(m): return 10.0 * (2.4 ** (m - 4.0))
print(f"events plotted: {len(lon)} | depth {dep.min():.0f}-{dep.max():.0f} km | mag {mag.min():.1f}-{mag.max():.1f}")

mpl.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(11.6, 8.6))
ax.imshow(rgb, extent=[W, E, S, N], origin="upper", zorder=0, aspect="auto",
          interpolation="bilinear")

_xs = W + (np.arange(z.shape[1])+0.5)*ds.res[0]
_ys = N - (np.arange(z.shape[0])+0.5)*ds.res[1]
ax.contour(_xs, _ys[::-1], np.flipud(z), levels=np.arange(1000,4001,500),
           colors="#7a4a1e", linewidths=0.35, alpha=0.55, zorder=1)

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
if lake:
    ax.add_collection(PatchCollection([Polygon(a, closed=True) for a in lake],
                      facecolor="#a9cfe7", edgecolor="#2f6d8f", linewidths=0.7,
                      alpha=0.55, zorder=1.5))

dnorm = Normalize(0, 50)
sc = ax.scatter(lon, lat, c=dep, s=msize(mag), cmap="plasma", norm=dnorm,
                edgecolors="black", linewidths=0.4, alpha=0.9, zorder=5)

halo = [pe.withStroke(linewidth=2.4, foreground="white")]
towns = [("Van",43.38,38.49,"left"),("Erci\u015f",43.36,39.03,"right"),
         ("\u00c7ald\u0131ran",43.90,39.14,"left"),("Muradiye",43.77,38.99,"left"),
         ("Edremit",43.24,38.42,"right"),("Ahlat",42.49,38.75,"right"),
         ("Adilcevaz",42.74,38.80,"right"),("Geva\u015f",43.11,38.29,"right"),
         ("G\u00fcrp\u0131nar",43.40,38.32,"left"),("\u00d6zalp",44.02,38.66,"left"),
         ("Ba\u015fkale",44.01,38.04,"left"),("Saray",44.15,38.68,"left"),
         ("Patnos",42.86,39.23,"right"),("Malazgirt",42.54,39.14,"right"),
         ("\u00c7atak",43.06,38.03,"right"),("Bah\u00e7esaray",42.80,38.13,"right")]
for nm, lo, la, ha in towns:
    if W < lo < E and S < la < N:
        ax.plot(lo, la, "s", ms=4.5, mfc="white", mec="black", mew=0.8, zorder=6)
        ax.text(lo + (0.028 if ha=="left" else -0.028), la, nm, fontsize=7.5,
                ha=ha, va="center", zorder=6, path_effects=halo)
ax.text(42.80, 38.60, "Lake Van", fontsize=14, style="italic", fontweight="bold",
        color="#00008b", ha="center", va="center", zorder=6,
        path_effects=[pe.withStroke(linewidth=3, foreground="white")])

principal = [(44.029,39.121,1976,7.3,"\u00c7ald\u0131ran",(43.72,39.32)),
             (43.508,38.721,2011,7.1,"Van",(43.99,38.95)),
             (43.229,38.429,2011,5.6,"Edremit",(43.62,38.26))]
for elo,ela,yr,mw,place,(tx,ty) in principal:
    ax.annotate(f"{yr}  M$_w${mw}  {place}", xy=(elo,ela), xytext=(tx,ty),
                fontsize=8.4, fontweight="bold", color="#111111", ha="center", va="center",
                zorder=8, path_effects=[pe.withStroke(linewidth=2.8, foreground="white")],
                arrowprops=dict(arrowstyle="-", color="#111111", lw=1.0,
                                path_effects=[pe.withStroke(linewidth=2.4, foreground="white")]))

ax.set_xlim(W, E); ax.set_ylim(S, N)
ax.set_aspect(1 / np.cos(np.deg2rad(MIDLAT)))
ax.set_xticks(np.arange(42.5, 44.31, 0.5)); ax.set_yticks(np.arange(38.0, 39.41, 0.5))
ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,_: f"{v:.1f}\u00b0E"))
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,_: f"{v:.1f}\u00b0N"))
ax.set_xticks(np.arange(42.3, 44.31, 0.1), minor=True)
ax.set_yticks(np.arange(38.0, 39.41, 0.1), minor=True)
ax.tick_params(which="both", direction="out", length=4, labelsize=9, top=True, right=True)
ax.grid(True, which="major", color="0.5", lw=0.4, alpha=0.35)
for s in ax.spines.values(): s.set_linewidth(1.1)
ax.set_title("Shaded-relief topography and seismicity of the "
             "\u00c7ald\u0131ran\u2013Erci\u015f\u2013Van region",
             fontsize=12.5, pad=10)

kmperdeg = 111.320 * np.cos(np.deg2rad(MIDLAT)); seg = 40.0 / kmperdeg
x0, y0 = E - 0.14 - seg, S + 0.10
ax.plot([x0, x0+seg], [y0, y0], "k-", lw=3, solid_capstyle="butt", zorder=7,
        path_effects=[pe.withStroke(linewidth=5, foreground="white")])
ax.text(x0+seg/2, y0+0.035, "40 km", ha="center", va="bottom", fontsize=8.5,
        zorder=7, path_effects=halo)

ax.annotate("N", xy=(E-0.12, N-0.08), xytext=(E-0.12, N-0.26), ha="center",
            va="center", fontsize=12, fontweight="bold", zorder=7, path_effects=halo,
            arrowprops=dict(arrowstyle="-|>", lw=2.2, color="black"))

from matplotlib.lines import Line2D
mags = [4, 5, 6, 7]
mh = [Line2D([0],[0], marker="o", linestyle="none", markersize=np.sqrt(msize(m)),
             markerfacecolor="0.75", markeredgecolor="black", markeredgewidth=0.5)
      for m in mags]
legm = ax.legend(mh, [f"M {m}" for m in mags], title="Magnitude", loc="lower left",
                 fontsize=8.5, title_fontsize=9, labelspacing=1.25, borderpad=0.9,
                 handletextpad=1.0, framealpha=0.9)
legm.get_frame().set_edgecolor("0.4"); ax.add_artist(legm)

cax_d = ax.inset_axes([1.03, 0.0, 0.028, 1.0])
cb_d = fig.colorbar(sc, cax=cax_d, extend="max")
cb_d.set_label("Focal depth (km)", fontsize=10)
cb_d.ax.invert_yaxis()
cax_e = ax.inset_axes([0.0, -0.10, 1.0, 0.03])
sm = SM(norm=Normalize(zmin, zmax), cmap=terr); sm.set_array([])
cb_e = fig.colorbar(sm, cax=cax_e, orientation="horizontal")
cb_e.set_label("Elevation (m)", fontsize=10)

ax.text(0.0, -0.20,
        "Data: GEBCO 2026 grid (topography); earthquakes from the IRIS Earthquake "
        "Browser (IEB), 1970\u20132026. Plotted with Matplotlib. Source: authors.",
        transform=ax.transAxes, fontsize=7.4, color="0.3", ha="left", va="top")

fig.savefig("outputs/fig03_dem_relief.pdf", bbox_inches="tight")
fig.savefig("outputs/fig03_dem_relief.png", dpi=600, bbox_inches="tight")
print("wrote fig03_dem_relief.pdf/.png")
