import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from matplotlib.cm import ScalarMappable as SM
from matplotlib.patches import Rectangle, Polygon, FancyArrow
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import rasterio, os

_SEARCH = ["data", ".", "grids", "data"]
def _ip(f):
    for d in _SEARCH:
        p = os.path.join(d, f)
        if os.path.exists(p):
            return p
    return f
_OUT = "." if os.path.isdir(".") else "outputs"
os.makedirs(_OUT, exist_ok=True)

MAIN  = _ip("region_relief.tif")
INSET = _ip("inset_relief.tif")
ARAB  = _ip("TP_Arabian.txt")
EURA  = _ip("TP_Eurasian.txt")

SW, SE, SS, SN = 42.3, 44.3, 38.0, 39.4

halo   = [pe.withStroke(linewidth=2.4, foreground="white")]
halo3  = [pe.withStroke(linewidth=3.0, foreground="white")]

def read_grid(path):
    ds = rasterio.open(path)
    z = ds.read(1).astype(float)
    if ds.nodata is not None:
        z[z == ds.nodata] = np.nan
    b = ds.bounds
    return z, (b.left, b.right, b.bottom, b.top), ds.res

z, (W, E, S, N), res = read_grid(MAIN)
MIDLAT = 0.5 * (S + N)
dy_m = res[1] * 111320.0
dx_m = res[0] * 111320.0 * np.cos(np.deg2rad(MIDLAT))
zland = np.where(z > 0, z, np.nan)
zmax  = np.nanmax(zland)

topo = plt.cm.turbo
ls = LightSource(azdeg=315, altdeg=45)
rgb = ls.shade(np.nan_to_num(z, nan=0.0), cmap=topo, blend_mode="soft",
               vert_exag=2.2, dx=dx_m, dy=dy_m, vmin=0, vmax=zmax)
sea = (z <= 0)
rgb[sea, :3] = np.array([0.686, 0.796, 0.882])
rgb[np.isnan(z), :3] = 1.0

def load_boundary(path, w, e, s, n, pad=1.0):
    lon, lat = [], []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        p = line.split()
        try:
            lo = float(p[0]); la = float(p[1])
        except (ValueError, IndexError):
            continue
        lo = ((lo + 180.0) % 360.0) - 180.0
        lon.append(lo); lat.append(la)
    lon = np.array(lon); lat = np.array(lat)
    out = (lon < w - pad) | (lon > e + pad) | (lat < s - pad) | (lat > n + pad)
    lon[out] = np.nan; lat[out] = np.nan
    return lon, lat

arab_lo, arab_la = load_boundary(ARAB, W, E, S, N)
eura_lo, eura_la = load_boundary(EURA, W, E, S, N)

NAF = [(34.0,40.75),(35.2,40.75),(36.4,40.70),(37.6,40.55),(38.6,40.35),
       (39.4,40.05),(40.1,39.75),(40.6,39.55),(41.05,39.30)]
EAF = [(41.05,39.30),(40.35,38.75),(39.55,38.30),(38.85,37.98),(38.15,37.68),
       (37.45,37.42),(36.85,37.18),(36.30,36.72),(36.12,36.35)]

LAKE_VAN = [(42.33,38.60),(42.55,38.71),(42.78,38.63),(42.98,38.73),(43.09,38.60),
            (43.04,38.44),(43.27,38.34),(43.40,38.41),(43.34,38.56),(43.14,38.66),
            (43.00,38.81),(42.79,38.87),(42.59,38.80),(42.44,38.72),(42.35,38.66)]
LAKE_URMIA = [(45.28,37.12),(45.55,37.22),(45.66,37.56),(45.60,37.92),(45.44,38.16),
              (45.29,38.10),(45.23,37.80),(45.19,37.44)]

mpl.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(12.2, 9.0))
ax.imshow(rgb, extent=[W, E, S, N], origin="upper", zorder=0, aspect="auto",
          interpolation="bilinear")

xs = np.linspace(W + res[0]/2, E - res[0]/2, z.shape[1])
ys = np.linspace(S + res[1]/2, N - res[1]/2, z.shape[0])
Zf = np.flipud(z)
ax.contour(xs, ys, Zf, levels=[1000, 2000, 3000],
           colors="#7a4a1e", linewidths=0.3, alpha=0.40, zorder=1)
ax.contour(xs, ys, Zf, levels=[0], colors="#1f4e6b", linewidths=0.6,
           alpha=0.8, zorder=1.2)

for poly, nm, txy in [(LAKE_VAN, "Lake Van", (42.72, 38.66)),
                      (LAKE_URMIA, "Lake\nUrmia", (45.42, 37.63))]:
    ax.add_patch(Polygon(poly, closed=True, facecolor="#a9cfe7",
                 edgecolor="#2f6d8f", linewidth=0.6, alpha=0.9, zorder=2.5))
ax.text(42.6, 38.72, "Lake Van", fontsize=9.5, style="italic", fontweight="bold",
        color="#00325c", ha="left", va="center", zorder=6, path_effects=halo)
ax.text(45.42, 37.63, "Lake\nUrmia", fontsize=7.6, style="italic",
        color="#00325c", ha="center", va="center", zorder=6, path_effects=halo)

for lo, la in [(arab_lo, arab_la), (eura_lo, eura_la)]:
    ax.plot(lo, la, "-", color="#b0161b", lw=3.4, alpha=0.95, zorder=3.4,
            solid_capstyle="round",
            path_effects=[pe.withStroke(linewidth=5.0, foreground="white")])

for trace in (NAF, EAF):
    xs_f = [p[0] for p in trace]; ys_f = [p[1] for p in trace]
    ax.plot(xs_f, ys_f, "-", color="#111111", lw=2.0, zorder=3.6,
            solid_capstyle="round",
            path_effects=[pe.withStroke(linewidth=3.4, foreground="white")])

ax.plot(41.05, 39.30, "*", ms=13, mfc="#ffd23f", mec="black", mew=0.8, zorder=6,
        path_effects=halo)

ax.text(39.0, 40.28, "North Anatolian Fault Zone", fontsize=9, rotation=-11,
        fontweight="bold", color="#111111", ha="center", va="bottom",
        zorder=7, path_effects=halo)
ax.text(38.1, 37.55, "East Anatolian\nFault Zone", fontsize=9, rotation=32,
        fontweight="bold", color="#111111", ha="center", va="center",
        zorder=7, path_effects=halo)
ax.text(44.7, 37.05, "Bitlis\u2013Zagros Suture Zone", fontsize=9, rotation=-50,
        fontweight="bold", color="#7a1013", ha="center", va="center",
        zorder=7, path_effects=halo)
ax.text(41.35, 39.55, "Karl\u0131ova\nTJ", fontsize=7.5, fontweight="bold",
        color="#5a4500", ha="left", va="bottom", zorder=7, path_effects=halo)

def plate(lo, la, name, col):
    ax.text(lo, la, name, fontsize=15, fontweight="bold", color=col,
            ha="center", va="center", zorder=7, alpha=0.9,
            path_effects=halo3)
plate(40.0, 42.35, "EURASIAN PLATE", "#20313f")
plate(35.4, 39.15, "ANATOLIAN\nPLATE", "#20313f")
plate(40.6, 34.05, "ARABIAN PLATE", "#20313f")

ax.annotate("", xy=(41.6, 36.4), xytext=(41.6, 34.9),
            arrowprops=dict(arrowstyle="-|>", lw=3.2, color="#0b3d0b",
            path_effects=[pe.withStroke(linewidth=5, foreground="white")]),
            zorder=7)
ax.text(41.85, 35.6, "Arabia\n~15\u201320 mm/yr", fontsize=7.8, color="#0b3d0b",
        ha="left", va="center", zorder=7, path_effects=halo)
ax.annotate("", xy=(34.6, 39.6), xytext=(36.4, 39.6),
            arrowprops=dict(arrowstyle="-|>", lw=3.2, color="#0b3d0b",
            path_effects=[pe.withStroke(linewidth=5, foreground="white")]),
            zorder=7)
ax.text(35.3, 39.85, "Anatolia \u2192 W\n~20\u201325 mm/yr", fontsize=7.8,
        color="#0b3d0b", ha="center", va="bottom", zorder=7, path_effects=halo)

ax.add_patch(Rectangle((SW, SS), SE-SW, SN-SS, fill=False, edgecolor="#d10000",
             linewidth=2.2, zorder=6,
             path_effects=[pe.withStroke(linewidth=3.6, foreground="white")]))
ax.text(0.5*(SW+SE), SN+0.10, "\u00c7ald\u0131ran\u2013Erci\u015f\u2013Van\nstudy area (Fig. 3)",
        fontsize=8.6, fontweight="bold", color="#8b0000", ha="center", va="bottom",
        zorder=7, path_effects=halo)

towns = [("Van",43.38,38.49,"left"),("Erzurum",41.28,39.91,"right"),
         ("Ta\u0163van",42.28,38.50,"right"),("A\u011fr\u0131",43.05,39.72,"left"),
         ("Hakk\u00e2ri",43.74,37.58,"left"),("Mu\u015f",41.49,38.75,"right"),
         ("Diyarbak\u0131r",40.23,37.91,"right"),("Tabriz",46.29,38.07,"left"),
         ("Bitlis",42.11,38.40,"right")]
for nm, lo, la, ha in towns:
    if W < lo < E and S < la < N:
        ax.plot(lo, la, "s", ms=4.0, mfc="white", mec="black", mew=0.7, zorder=6)
        ax.text(lo + (0.10 if ha=="left" else -0.10), la, nm, fontsize=7.4,
                ha=ha, va="center", zorder=6, path_effects=halo)

ax.set_xlim(W, E); ax.set_ylim(S, N)
ax.set_aspect(1 / np.cos(np.deg2rad(MIDLAT)))
ax.set_xticks(np.arange(34, 48.1, 2))
ax.set_yticks(np.arange(34, 43.1, 2))
ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,_: f"{v:.0f}\u00b0E"))
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,_: f"{v:.0f}\u00b0N"))
ax.set_xticks(np.arange(33, 48.1, 1), minor=True)
ax.set_yticks(np.arange(33, 43.1, 1), minor=True)
ax.tick_params(which="both", direction="out", length=4, labelsize=9,
               top=True, right=True)
ax.grid(True, which="major", color="white", lw=0.5, alpha=0.55)
for s in ax.spines.values():
    s.set_linewidth(1.1)
ax.set_title("Regional tectonic setting of Eastern T\u00fcrkiye \u2014 "
             "Arabia\u2013Eurasia collision zone", fontsize=13, pad=10)

kmperdeg = 111.320 * np.cos(np.deg2rad(MIDLAT)); seg = 200.0 / kmperdeg
x0, y0 = E - 0.4 - seg, S + 0.35
ax.plot([x0, x0+seg], [y0, y0], "k-", lw=3, solid_capstyle="butt", zorder=7,
        path_effects=[pe.withStroke(linewidth=5, foreground="white")])
ax.text(x0+seg/2, y0+0.14, "200 km", ha="center", va="bottom", fontsize=8.5,
        zorder=7, path_effects=halo)

ax.annotate("N", xy=(W+0.55, S+1.55), xytext=(W+0.55, S+0.55), ha="center",
            va="center", fontsize=13, fontweight="bold", zorder=7,
            path_effects=halo,
            arrowprops=dict(arrowstyle="-|>", lw=2.4, color="black"))

handles = [Line2D([0],[0], color="#b0161b", lw=3.2),
           Line2D([0],[0], color="#111111", lw=2.0),
           Line2D([0],[0], marker="*", linestyle="none", markersize=12,
                  markerfacecolor="#ffd23f", markeredgecolor="black"),
           Line2D([0],[0], color="#d10000", lw=2.2)]
labels = ["Plate boundary (Arabia, Eurasia)", "Major fault zone (schematic)",
          "Karl\u0131ova triple junction", "Study area (Fig. 3)"]
leg = ax.legend(handles, labels, loc="upper left", fontsize=8.0, framealpha=0.92,
                borderpad=0.7, handletextpad=0.9, labelspacing=0.7)
leg.get_frame().set_edgecolor("0.4"); ax.add_artist(leg)

cax = ax.inset_axes([1.025, 0.0, 0.026, 1.0])
sm = SM(norm=Normalize(0, zmax), cmap=topo); sm.set_array([])
cb = fig.colorbar(sm, cax=cax, extend="max")
cb.set_label("Elevation (m)", fontsize=10)

zi, (iW, iE, iS, iN), ires = read_grid(INSET)
ils = LightSource(azdeg=315, altdeg=45)
gray = ils.hillshade(np.nan_to_num(zi, nan=0.0), vert_exag=1.0,
                     dx=ires[0]*111320*np.cos(np.deg2rad(0.5*(iS+iN))),
                     dy=ires[1]*111320)
irgb = np.dstack([gray]*3)
land = zi > 0
base = np.empty(zi.shape + (3,))
base[land] = (0.80, 0.78, 0.72); base[~land] = (0.72, 0.82, 0.90)
irgb = 0.55*irgb + 0.45*base
axin = ax.inset_axes([0.700, 0.655, 0.290, 0.335])
axin.imshow(np.clip(irgb, 0, 1), extent=[iW, iE, iS, iN], origin="upper",
            aspect=1/np.cos(np.deg2rad(0.5*(iS+iN))))
axin.add_patch(Rectangle((W, S), E-W, N-S, fill=False, edgecolor="#d10000",
               lw=1.6))
axin.set_xticks([]); axin.set_yticks([])
for s in axin.spines.values():
    s.set_linewidth(1.0); s.set_edgecolor("0.3")
axin.text(0.5, 1.02, "Location", transform=axin.transAxes, fontsize=8,
          ha="center", va="bottom", fontweight="bold")

def wrap_to_axes(axes, s, fontsize, frac=0.99):
    from matplotlib.font_manager import FontProperties
    fig = axes.figure; fig.canvas.draw()
    r = fig.canvas.get_renderer()
    axpx = axes.get_window_extent(renderer=r).width * frac
    fp = FontProperties(family="DejaVu Sans", size=fontsize)
    lines, cur = [], ""
    for w in s.split():
        trial = w if not cur else cur + " " + w
        if r.get_text_width_height_descent(trial, fp, False)[0] <= axpx or not cur:
            cur = trial
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)

_credit = ("Data: GEBCO 2026 / SRTM15+ (GMT earth_relief); plate boundaries "
           "(Arabian, Eurasian); North/East Anatolian Fault Zones and "
           "Bitlis\u2013Zagros Suture. Plotted with Matplotlib (Python). Source: authors.")
ax.text(0.0, -0.075, wrap_to_axes(ax, _credit, 7.0),
        transform=ax.transAxes, fontsize=7.0, color="0.3", ha="left", va="top")

fig.savefig(os.path.join(_OUT, "fig01_regional_setting.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(_OUT, "fig01_regional_setting.png"), dpi=200,
            bbox_inches="tight")
print("events/frame: W,E,S,N =", W, E, S, N, "zmax land =", round(float(zmax)))
print("wrote fig01_regional_setting.pdf/.png")
