import csv
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from matplotlib.cm import ScalarMappable as SM
from matplotlib.patches import Polygon, FancyArrow
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
MID = 0.5 * (S + N)
COSL = np.cos(np.deg2rad(MID))
dx = ds.res[0] * 111320 * COSL
dy = ds.res[1] * 111320
extent = [W, E, S, N]

lon1d = W + (np.arange(z.shape[1]) + 0.5) * ds.res[0]
lat1d = N - (np.arange(z.shape[0]) + 0.5) * ds.res[1]
LON, LAT = np.meshgrid(lon1d, lat1d)

gy, gx = np.gradient(np.nan_to_num(z), dy, dx)
slope = np.degrees(np.arctan(np.hypot(gx, gy)))
relief = (ndimage.maximum_filter(np.nan_to_num(z), 5)
          - ndimage.minimum_filter(np.nan_to_num(z), 5))

LAKE_LVL = 1648.0
lowm = z <= LAKE_LVL
lab, nlab = ndimage.label(lowm)
sizes = ndimage.sum(np.ones_like(lab), lab, range(1, nlab + 1))
lake = lab == (1 + int(np.argmax(sizes)))
lake = ndimage.binary_closing(lake, iterations=1)

shore_dist_px = ndimage.distance_transform_edt(~lake)
shore_km = shore_dist_px * dx / 1000.0

VANF = np.array([(43.18, 38.60), (43.34, 38.66),
                 (43.50, 38.72), (43.64, 38.79)])
def dist_to_polyline(LON, LAT, pts):
    d = np.full(LON.shape, 1e9)
    for i in range(len(pts) - 1):
        ax_, ay_ = pts[i]; bx_, by_ = pts[i + 1]
        vx = (bx_ - ax_) * COSL; vy = (by_ - ay_); L2 = vx * vx + vy * vy
        wx = (LON - ax_) * COSL; wy = (LAT - ay_)
        t = np.clip((wx * vx + wy * vy) / L2, 0, 1)
        d = np.minimum(d, np.hypot(wx - t * vx, wy - t * vy) * 111.0)
    return d
distf = dist_to_polyline(LON, LAT, VANF)

flat = np.clip(1.0 - slope / 8.0, 0, 1) ** 1.3
hal = z - LAKE_LVL
gw = np.exp(-np.clip(hal, 0, None) / 90.0)
shore = np.exp(-shore_km / 9.0)
Liq = flat * (0.55 * gw + 0.45 * shore)

Slide = 0.62 * np.clip(slope / 28.0, 0, 1) + 0.38 * np.clip(relief / 420.0, 0, 1)

prox = 0.35 + 0.65 * np.exp(-distf / 26.0)
Sr = prox * (0.70 * Liq + 0.55 * Slide)
Sr[lake] = np.nan
Sr = Sr / np.nanpercentile(Sr, 99.3)
Sr = np.clip(Sr, 0, 1)

eq2011 = []
mainshock = None
with open(EQ, newline="") as f:
    for r in csv.DictReader(f):
        try:
            lat = float(r["Lat"]); lon = float(r["Lon"])
            mag = float(r["Mag"]); yr = int(float(r["Year"]))
        except ValueError:
            continue
        if not (W <= lon <= E and S <= lat <= N):
            continue
        if yr == 2011:
            eq2011.append((lon, lat, mag))
            if mag >= 7.0:
                mainshock = (lon, lat, mag)
eq2011 = np.array(eq2011)

TOWNS = {"Ercis": (43.360, 39.026, "right"),
         "Van":   (43.380, 38.494, "right"),
         "Edremit": (43.245, 38.424, "left")}

CORR = [(43.36, 38.52), (43.55, 38.74), (43.58, 38.95),
        (43.44, 39.04), (43.28, 38.98), (43.24, 38.74), (43.30, 38.55)]

fig, ax = plt.subplots(figsize=(8.4, 6.7))
ax.set_facecolor("white")
ls = LightSource(azdeg=315, altdeg=45)
hs = ls.hillshade(np.nan_to_num(z, nan=np.nanmin(z)), vert_exag=1.6,
                  dx=dx, dy=dy)
ax.imshow(hs, extent=extent, cmap="gray", vmin=0.15, vmax=1.05,
          origin="upper", interpolation="bilinear", zorder=1)

cmap = plt.get_cmap("YlOrRd").copy()
im = ax.imshow(np.ma.masked_invalid(Sr), extent=extent, cmap=cmap,
               norm=Normalize(0, 1), origin="upper", alpha=0.72,
               interpolation="bilinear", zorder=2)

z_land = np.where(lake, np.nan, z)
levels = np.arange(1000, 4001, 500)
ax.contour(LON, LAT, z_land, levels=levels, colors="#7a4a1e",
           linewidths=0.4, alpha=0.7, zorder=2.4)

lake_rgba = np.zeros((*lake.shape, 4))
lake_rgba[lake] = mpl.colors.to_rgba("#9dc3e6")
ax.imshow(lake_rgba, extent=extent, origin="upper", zorder=3,
          interpolation="nearest")

ax.contour(LON, LAT, lake.astype(float), levels=[0.5],
           colors="#2b5d8a", linewidths=0.7, zorder=3.3)
ax.text(42.92, 38.64, "Lake\nVan", ha="center", va="center", zorder=6,
        fontsize=11, style="italic", color="white", fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2.4, foreground="#173d5c")])

ax.add_patch(Polygon(CORR, closed=True, facecolor="#3a2a10",
                     edgecolor="#3a2a10", lw=1.4, ls=(0, (5, 2)),
                     alpha=0.10, zorder=3.7))
ax.add_patch(Polygon(CORR, closed=True, facecolor="none",
                     edgecolor="#3a2a10", lw=1.4, ls=(0, (5, 2)),
                     zorder=4.5))
ax.plot([], [], color="#3a2a10", lw=1.4, ls=(0, (5, 2)),
        label="Reported 2011 ground-failure corridor")

vf = np.array(VANF)
ax.plot(vf[:, 0], vf[:, 1], color="black", lw=2.0, zorder=5,
        solid_capstyle="round")
for i in range(len(vf) - 1):
    mx, my = (vf[i] + vf[i + 1]) / 2
    ang = np.arctan2((vf[i + 1][1] - vf[i][1]),
                     (vf[i + 1][0] - vf[i][0]) * COSL)
    nx, ny = -np.sin(ang) * 0.045 / COSL, np.cos(ang) * 0.045
    ax.add_patch(Polygon([(mx, my), (mx + nx * 0.6 - np.cos(ang) * 0.03,
                                     my + ny * 0.6 - np.sin(ang) * 0.03),
                          (mx + nx * 0.6 + np.cos(ang) * 0.03,
                           my + ny * 0.6 + np.sin(ang) * 0.03)],
                         closed=True, facecolor="black", edgecolor="black",
                         zorder=5))
ax.text(43.30, 38.60, "Van Fault", rotation=22, fontsize=8.5,
        color="black", zorder=6, ha="center", va="top", fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

if eq2011.size:
    ax.scatter(eq2011[:, 0], eq2011[:, 1], s=16, marker="o",
               facecolor="#1f4e79", edgecolor="white", linewidth=0.4,
               alpha=0.85, zorder=6, label="2011 aftershocks ($M\\geq4$)")

if mainshock:
    ax.scatter([mainshock[0]], [mainshock[1]], s=360, marker="*",
               facecolor="#ffe000", edgecolor="black", linewidth=1.3,
               zorder=8, label="2011 Van-Erci\u015f mainshock ($M_w$ 7.1)")

for name, (lon, lat, ha) in TOWNS.items():
    ax.plot(lon, lat, "s", ms=6.5, mfc="white", mec="black", mew=1.3,
            zorder=7)
    off = 0.03 if ha == "left" else -0.03
    ax.text(lon + off, lat, name.replace("Ercis", "Erci\u015f"),
            fontsize=9, ha=ha, va="center", zorder=7, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])

ax.set_xlim(W, E); ax.set_ylim(S, N)
ax.set_aspect(1.0 / COSL)
ax.set_xticks(np.arange(42.5, 44.4, 0.5))
ax.set_yticks(np.arange(38.0, 39.5, 0.5))
ax.set_xticklabels([f"{v:.1f}\u00b0E" for v in np.arange(42.5, 44.4, 0.5)])
ax.set_yticklabels([f"{v:.1f}\u00b0N" for v in np.arange(38.0, 39.5, 0.5)])

ax.xaxis.set_minor_locator(MultipleLocator(1.0 / 60.0))
ax.yaxis.set_minor_locator(MultipleLocator(1.0 / 60.0))
ax.tick_params(axis="both", which="major", direction="out", length=5,
               width=1.0, labelsize=8.5)
ax.tick_params(axis="both", which="minor", direction="out", length=2.5,
               width=0.6, color="0.25")
for gx_ in np.arange(42.5, 44.4, 0.5):
    ax.axvline(gx_, color="white", lw=0.5, alpha=0.55, zorder=3.6)
for gy_ in np.arange(38.0, 39.5, 0.5):
    ax.axhline(gy_, color="white", lw=0.5, alpha=0.55, zorder=3.6)
for s in ax.spines.values():
    s.set_linewidth(1.1)

km = 40.0
dlon = km / (111.320 * COSL)
x0 = W + 0.06 * (E - W); y0 = S + 0.065 * (N - S)
ax.plot([x0, x0 + dlon], [y0, y0], color="black", lw=3.2,
        solid_capstyle="butt", zorder=9)
ax.text(x0 + dlon / 2, y0 + 0.018 * (N - S), f"{int(km)} km", ha="center",
        va="bottom", fontsize=8, zorder=9,
        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

xn = E - 0.055 * (E - W); yn = S + 0.10 * (N - S)
ax.annotate("N", xy=(xn, yn + 0.075 * (N - S)), xytext=(xn, yn),
            ha="center", va="center", fontsize=9, fontweight="bold",
            zorder=9, arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8),
            path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])

handles, labels = ax.get_legend_handles_labels()
order = ["2011 Van-Erci\u015f mainshock ($M_w$ 7.1)",
         "2011 aftershocks ($M\\geq4$)",
         "Reported 2011 ground-failure corridor"]
hd = dict(zip(labels, handles))
handles = [hd[o] for o in order if o in hd]
handles.append(Line2D([0], [0], color="black", lw=2.0,
                      label="Van Fault (thrust)"))
handles.append(Line2D([0], [0], marker="s", color="none", mfc="white",
                      mec="black", mew=1.3, ms=6.5, label="Town"))
leg = ax.legend(handles=handles, loc="upper left", fontsize=9,
                framealpha=0.92, edgecolor="0.5", borderpad=0.6,
                handletextpad=0.6)
leg.set_zorder(10)

ax.set_title("Cascade\u2009II: liquefaction, lateral-spreading and landslide "
             "susceptibility\nVan\u2013Erci\u015f corridor "
             "(2011 $M_w$\u20097.1 Van earthquake)", fontsize=11,
             fontweight="bold", pad=8)

fig.subplots_adjust(left=0.075, right=0.865, top=0.90, bottom=0.170)
fig.canvas.draw()
pos = ax.get_position()
cax = fig.add_axes([pos.x1 + 0.018, pos.y0, 0.020, pos.height])
cb = fig.colorbar(SM(norm=Normalize(0, 1), cmap=cmap), cax=cax)
cb.set_label("Cascade\u2009II susceptibility (probability)", fontsize=9)
cb.ax.yaxis.set_major_locator(MultipleLocator(0.2))
cb.ax.yaxis.set_minor_locator(AutoMinorLocator(4))
cb.ax.tick_params(which="major", labelsize=8, length=4)
cb.ax.tick_params(which="minor", length=2.5)

CRED_FS = 8
segments = [
    "Depiction: liquefaction, lateral-spreading and landslide susceptibility "
    "(0\u20131) from terrain and Van-fault / Lake-Van-shore proximity; Van Fault; "
    "2011 mainshock and aftershocks ($M\\geq4$); reported ground-failure "
    "corridor; 500 m elevation contours.",
    "Data: GEBCO 2026 (15\u2033); EarthScope IEB catalogue (IRIS/USGS).  "
    "Software: Python \u2014 rasterio 1.5, matplotlib 3.10, numpy 2.4, "
    "scipy 1.17.  Source: authors.",
]
rnd = fig.canvas.get_renderer()
fig_w_px = fig.get_size_inches()[0] * fig.dpi
max_px = 0.98 * pos.width * fig_w_px
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
fig.text((pos.x0 + pos.x1) / 2, 0.014, "\n".join(lines), ha="center",
         va="bottom", fontsize=CRED_FS, color="0.20", linespacing=1.35)

fig.savefig("fig11_cascade2_liquefaction.png", dpi=200)
fig.savefig("fig11_cascade2_liquefaction_hi.png", dpi=600)
fig.savefig("fig11_cascade2_liquefaction.pdf")
print("saved; mainshock =", mainshock, "| n_2011 =", len(eq2011))
print("map box:", [round(v, 3) for v in (pos.x0, pos.y0, pos.width, pos.height)])
