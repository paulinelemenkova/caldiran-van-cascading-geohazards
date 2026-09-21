import csv
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from matplotlib.cm import ScalarMappable as SM
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import matplotlib.patheffects as pe
import rasterio
from scipy import ndimage

DEM = "gebco_2026_n39_4_s38_0_w42_3_e44_3_geotiff.tif"
EQ  = "IEB_Turkey_5000_events.csv"

ds = rasterio.open(DEM)
z = ds.read(1).astype(float)
if ds.nodata is not None:
    z[z == ds.nodata] = np.nan
W, E, S, N = ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top
MID = 0.5 * (S + N); COSL = np.cos(np.deg2rad(MID))
dx = ds.res[0] * 111320 * COSL; dy = ds.res[1] * 111320
extent = [W, E, S, N]
lon1d = W + (np.arange(z.shape[1]) + 0.5) * ds.res[0]
lat1d = N - (np.arange(z.shape[0]) + 0.5) * ds.res[1]
LON, LAT = np.meshgrid(lon1d, lat1d)

gy, gx = np.gradient(np.nan_to_num(z), dy, dx)
slope = np.degrees(np.arctan(np.hypot(gx, gy)))
lk = ndimage.label(z <= 1648.0)[0]
lake = lk == (1 + int(np.argmax(ndimage.sum(np.ones_like(lk), lk, range(1, lk.max() + 1)))))
lake = ndimage.binary_closing(lake, iterations=1)

dist_px, (ir, ic) = ndimage.distance_transform_edt(lake, return_indices=True)
margin_slope = slope[ir, ic]
shore_km = dist_px * dx / 1000.0

VANF = np.array([(43.18, 38.60), (43.34, 38.66), (43.50, 38.72), (43.64, 38.79)])
def dist_to_polyline(LON, LAT, pts):
    d = np.full(LON.shape, 1e9)
    for i in range(len(pts) - 1):
        ax_, ay_ = pts[i]; bx_, by_ = pts[i + 1]
        vx = (bx_ - ax_) * COSL; vy = (by_ - ay_); L2 = vx * vx + vy * vy
        wx = (LON - ax_) * COSL; wy = (LAT - ay_)
        t = np.clip((wx * vx + wy * vy) / L2, 0, 1)
        d = np.minimum(d, np.hypot(wx - t * vx, wy - t * vy) * 111.0)
    return d
fault_km = dist_to_polyline(LON, LAT, VANF)

mslope = np.clip(margin_slope / 15.0, 0, 1) ** 0.9
shorew = np.exp(-shore_km / 6.0)
faultf = 0.65 + 0.35 * np.exp(-fault_km / 25.0)
Sr = mslope * (0.35 + 0.65 * shorew) * faultf
Sr = np.where(lake, Sr, np.nan)
Sr = Sr / np.nanpercentile(Sr, 99.0)
Sr = np.clip(Sr, 0, 1)

eq2011 = []
mainshock = None
with open(EQ, newline="") as f:
    for r in csv.DictReader(f):
        try:
            la = float(r["Lat"]); lo = float(r["Lon"])
            mg = float(r["Mag"]); yr = int(float(r["Year"]))
        except ValueError:
            continue
        if W <= lo <= E and S <= la <= N and yr == 2011:
            eq2011.append((lo, la, mg))
            if mg >= 7.0:
                mainshock = (lo, la, mg)
eq2011 = np.array(eq2011)

TOWNS = {"Ercis": (43.360, 39.026, "right"), "Van": (43.380, 38.494, "right"),
         "Ahlat": (42.489, 38.752, "left"), "Geva\u015f": (43.10, 38.30, "left")}

fig, ax = plt.subplots(figsize=(8.4, 6.7))
ls = LightSource(azdeg=315, altdeg=45)
hs = ls.hillshade(np.nan_to_num(z, nan=np.nanmin(z)), vert_exag=1.6, dx=dx, dy=dy)
ax.imshow(hs, extent=extent, cmap="gray", vmin=0.15, vmax=1.05, origin="upper",
          interpolation="bilinear", zorder=1)

cmap = plt.get_cmap("YlOrRd").copy()
ax.imshow(np.ma.masked_invalid(Sr), extent=extent, cmap=cmap,
          norm=Normalize(0, 1), origin="upper", alpha=0.85,
          interpolation="bilinear", zorder=2)

ax.contour(LON, LAT, lake.astype(float), levels=[0.5], colors="#22506f",
           linewidths=1.0, zorder=3.3)
ax.text(42.60, 38.42, "Lake Van", ha="center", va="center", zorder=6,
        fontsize=11, style="italic", color="white", fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2.4, foreground="#173d5c")])

vf = VANF
ax.plot(vf[:, 0], vf[:, 1], color="black", lw=2.0, zorder=5, solid_capstyle="round")
for i in range(len(vf) - 1):
    mx, my = (vf[i] + vf[i + 1]) / 2
    ang = np.arctan2((vf[i + 1][1] - vf[i][1]), (vf[i + 1][0] - vf[i][0]) * COSL)
    nx, ny = -np.sin(ang) * 0.045 / COSL, np.cos(ang) * 0.045
    ax.add_patch(plt.Polygon([(mx, my),
                              (mx + nx * 0.6 - np.cos(ang) * 0.03, my + ny * 0.6 - np.sin(ang) * 0.03),
                              (mx + nx * 0.6 + np.cos(ang) * 0.03, my + ny * 0.6 + np.sin(ang) * 0.03)],
                             closed=True, facecolor="black", zorder=5))
ax.text(43.34, 38.60, "Van Fault", rotation=22, fontsize=8.5, color="black",
        zorder=6, ha="center", va="top", fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

if eq2011.size:
    ax.scatter(eq2011[:, 0], eq2011[:, 1], s=8, marker="o", facecolor="#1f4e79",
               edgecolor="none", alpha=0.5, zorder=5.5,
               label="2011 seismicity ($M\\geq4$)")
if mainshock:
    ax.scatter([mainshock[0]], [mainshock[1]], s=300, marker="*",
               facecolor="#ffe000", edgecolor="black", linewidth=1.2, zorder=8,
               label="2011 Van-Erci\u015f mainshock ($M_w$ 7.1)")

for name, (lo, la, ha) in TOWNS.items():
    ax.plot(lo, la, "s", ms=6.0, mfc="white", mec="black", mew=1.2, zorder=7)
    off = 0.03 if ha == "left" else -0.03
    ax.text(lo + off, la, name, fontsize=9, ha=ha, va="center", zorder=7,
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])

ax.set_xlim(W, E); ax.set_ylim(S, N); ax.set_aspect(1.0 / COSL)
majx = np.arange(42.5, 44.4, 0.5); majy = np.arange(38.0, 39.5, 0.5)
ax.set_xticks(majx); ax.set_yticks(majy)
ax.set_xticklabels([f"{v:.1f}\u00b0E" for v in majx])
ax.set_yticklabels([f"{v:.1f}\u00b0N" for v in majy])
ax.xaxis.set_minor_locator(MultipleLocator(1.0 / 60.0))
ax.yaxis.set_minor_locator(MultipleLocator(1.0 / 60.0))
ax.tick_params(which="major", direction="out", length=5, width=1.0, labelsize=8.5)
ax.tick_params(which="minor", direction="out", length=2.5, width=0.6, color="0.25")
for gxx in majx:
    ax.axvline(gxx, color="white", lw=0.75, alpha=0.6, zorder=3.6)
for gyy in majy:
    ax.axhline(gyy, color="white", lw=0.75, alpha=0.6, zorder=3.6)
for sp in ax.spines.values():
    sp.set_linewidth(1.1)

kmv = 40.0; dlon = kmv / (111.320 * COSL)
x0 = W + 0.06 * (E - W); y0 = S + 0.065 * (N - S)
ax.plot([x0, x0 + dlon], [y0, y0], color="black", lw=3.2, solid_capstyle="butt", zorder=9)
ax.text(x0 + dlon / 2, y0 + 0.018 * (N - S), f"{int(kmv)} km", ha="center",
        va="bottom", fontsize=8, zorder=9,
        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
xn = E - 0.055 * (E - W); yn = S + 0.10 * (N - S)
ax.annotate("N", xy=(xn, yn + 0.075 * (N - S)), xytext=(xn, yn), ha="center",
            va="center", fontsize=9, fontweight="bold", zorder=9,
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8),
            path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])

handles = [
    Line2D([0], [0], marker="*", color="none", mfc="#ffe000", mec="black",
           mew=1.1, ms=14, label="2011 Van-Erci\u015f mainshock ($M_w$ 7.1)"),
    Line2D([0], [0], marker="o", color="none", mfc="#1f4e79", mec="none", ms=6,
           alpha=0.6, label="2011 seismicity ($M\\geq4$)"),
    Line2D([0], [0], color="black", lw=2.0, label="Van Fault (thrust)"),
    Line2D([0], [0], marker="s", color="none", mfc="white", mec="black",
           mew=1.2, ms=6, label="Town"),
]
leg = ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.92,
                edgecolor="0.5", borderpad=0.6, handletextpad=0.6)
leg.set_zorder(10)

ax.set_title("Cascade\u2009IV: sublacustrine mass-wasting susceptibility, "
             "Lake Van floor\nindex from lake-margin steepness and "
             "active-fault proximity", fontsize=11, fontweight="bold", pad=8)

fig.subplots_adjust(left=0.075, right=0.865, top=0.895, bottom=0.180)
fig.canvas.draw()
pos = ax.get_position()
cax = fig.add_axes([pos.x1 + 0.018, pos.y0, 0.020, pos.height])
cb = fig.colorbar(SM(norm=Normalize(0, 1), cmap=cmap), cax=cax)
cb.set_label("Cascade\u2009IV susceptibility (relative)", fontsize=9)
cb.ax.yaxis.set_major_locator(MultipleLocator(0.2))
cb.ax.yaxis.set_minor_locator(AutoMinorLocator(4))
cb.ax.tick_params(which="major", labelsize=8, length=4)
cb.ax.tick_params(which="minor", length=2.5)

CRED_FS = 8
segments = [
    "Depiction: sublacustrine mass-wasting susceptibility (relative) on the "
    "Lake Van floor from lake-margin terrain steepness, shore-margin proximity "
    "and Van-Fault proximity; Van Fault; 2011 mainshock and seismicity "
    "($M\\geq4$); grey hillshade on land.",
    "Data: GEBCO 2026 (15\u2033); EarthScope IEB (IRIS/USGS).  Software: Python "
    "(rasterio, matplotlib, numpy, scipy).  Source: authors.",
]
rnd = fig.canvas.get_renderer()
max_px = 0.98 * pos.width * fig.get_size_inches()[0] * fig.dpi
probe = fig.text(0, 0, "", fontsize=CRED_FS)
lines = []
for seg in segments:
    cur = ""
    for word in seg.split(" "):
        trial = (cur + " " + word).strip()
        probe.set_text(trial)
        if probe.get_window_extent(rnd).width <= max_px or not cur:
            cur = trial
        else:
            lines.append(cur); cur = word
    if cur:
        lines.append(cur)
probe.remove()
fig.text((pos.x0 + pos.x1) / 2, 0.013, "\n".join(lines), ha="center",
         va="bottom", fontsize=CRED_FS, color="0.20", linespacing=1.32)

fig.savefig("fig13_cascade4_lake_van.png", dpi=200)
fig.savefig("fig13_cascade4_lake_van_hi.png", dpi=600)
fig.savefig("fig13_cascade4_lake_van.pdf")
print("saved | lake cells=%d | Sr max=%.2f | credit lines=%d"
      % (int(lake.sum()), float(np.nanmax(Sr)), len(lines)))
