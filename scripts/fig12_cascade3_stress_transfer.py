import csv
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, TwoSlopeNorm
from matplotlib.cm import ScalarMappable as SM
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import matplotlib.patheffects as pe
import rasterio
from scipy import ndimage
from okada_wrapper import dc3dwrapper

DEM = "gebco_2026_n39_4_s38_0_w42_3_e44_3_geotiff.tif"
EQ  = "IEB_Turkey_5000_events.csv"

mu = 3.3e10; lam = mu
alpha = (lam + mu) / (lam + 2 * mu)
mu_f = 0.4
EPI_LON, EPI_LAT = 43.508, 38.721
strike, dip, rake = 246.0, 42.0, 85.0
L, Wd, top_depth, U = 35.0, 18.0, 4.0, 2.7
cen_depth = top_depth + 0.5 * Wd * np.sin(np.radians(dip))
r_strike, r_dip, r_rake = 246.0, 42.0, 85.0
Z_OBS = 7.0
VE_LON, VE_LAT = 43.229, 38.429

def enu(lon, lat):
    k = 111.320
    return ((lon - EPI_LON) * k * np.cos(np.radians(EPI_LAT)),
            (lat - EPI_LAT) * k)

def plane_vectors(sd, dd, rd):
    phi, dl, rk = map(np.radians, (sd, dd, rd))
    s = np.array([np.sin(phi), np.cos(phi), 0.0])
    h = np.array([np.cos(phi), -np.sin(phi), 0.0])
    d = np.cos(dl) * h - np.sin(dl) * np.array([0, 0, 1.0])
    n = np.array([np.sin(dl) * np.cos(phi), -np.sin(dl) * np.sin(phi), np.cos(dl)])
    u = np.cos(rk) * s + np.sin(rk) * (-d)
    return s, h, n, u

def dcfs_field(LON, LAT):
    s, h, _, _ = plane_vectors(strike, dip, rake)
    R = np.array([s, h, [0, 0, 1.0]]).T
    ss, ds = U * np.cos(np.radians(rake)), U * np.sin(np.radians(rake))
    _, _, nr, ur = plane_vectors(r_strike, r_dip, r_rake)
    out = np.full(LON.shape, np.nan)
    for j in range(LON.shape[0]):
        for i in range(LON.shape[1]):
            xe, yn = enu(LON[j, i], LAT[j, i])
            xf, yf = xe * s[0] + yn * s[1], xe * h[0] + yn * h[1]
            ok, _, grad = dc3dwrapper(alpha, [xf, yf, -Z_OBS], cen_depth, dip,
                                      [-L / 2, L / 2], [-Wd / 2, Wd / 2],
                                      [ss, ds, 0.0])
            if ok != 0:
                continue
            g = np.array(grad)
            e = 0.5 * (g + g.T)
            sig = R @ (lam * np.trace(e) * np.eye(3) + 2 * mu * e) @ R.T
            t = sig @ nr
            out[j, i] = (ur @ t + mu_f * (nr @ t))
    return out / 1e3 / 1e6

def fault_footprint():
    s, h, _, _ = plane_vectors(strike, dip, rake)
    hw = 0.5 * Wd * np.cos(np.radians(dip))
    top = -hw * h; bot = +hw * h
    corners = [top - L / 2 * s, top + L / 2 * s, bot + L / 2 * s, bot - L / 2 * s]
    def to_ll(v):
        k = 111.320
        return (EPI_LON + v[0] / (k * np.cos(np.radians(EPI_LAT))),
                EPI_LAT + v[1] / k)
    poly = [to_ll(c) for c in corners]
    trace = [to_ll(top - L / 2 * s), to_ll(top + L / 2 * s)]
    return poly, trace, h

ds = rasterio.open(DEM)
z = ds.read(1).astype(float)
if ds.nodata is not None:
    z[z == ds.nodata] = np.nan
W, E, S, N = ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top
MID = 0.5 * (S + N); COSL = np.cos(np.deg2rad(MID))
dx = ds.res[0] * 111320 * COSL; dy = ds.res[1] * 111320
extent = [W, E, S, N]

lowm = z <= 1648.0
lab, nl = ndimage.label(lowm)
lake = lab == (1 + int(np.argmax(ndimage.sum(np.ones_like(lab), lab, range(1, nl + 1)))))

gx = np.linspace(W, E, 150); gy = np.linspace(N, S, 105)
GX, GY = np.meshgrid(gx, gy)
CFS = dcfs_field(GX, GY)
ve_val = dcfs_field(np.array([[VE_LON]]), np.array([[VE_LAT]]))[0, 0]

eq2011 = []
with open(EQ, newline="") as f:
    for r in csv.DictReader(f):
        try:
            lat = float(r["Lat"]); lon = float(r["Lon"])
            mag = float(r["Mag"]); yr = int(float(r["Year"]))
        except ValueError:
            continue
        if W <= lon <= E and S <= lat <= N and yr == 2011:
            eq2011.append((lon, lat, mag))
eq2011 = np.array(eq2011)

TOWNS = {"Ercis": (43.360, 39.026, "right"), "Van": (43.380, 38.494, "right"),
         "Edremit": (43.245, 38.424, "left")}
poly, trace, hvec = fault_footprint()

fig, ax = plt.subplots(figsize=(8.4, 6.7))
ls = LightSource(azdeg=315, altdeg=45)
hs = ls.hillshade(np.nan_to_num(z, nan=np.nanmin(z)), vert_exag=1.5, dx=dx, dy=dy)
ax.imshow(hs, extent=extent, cmap="gray", vmin=0.25, vmax=1.1,
          origin="upper", interpolation="bilinear", zorder=1)

CLIP = 0.02
cmap = plt.get_cmap("RdBu_r").copy()
norm = TwoSlopeNorm(vmin=-CLIP, vcenter=0.0, vmax=CLIP)
im = ax.imshow(np.clip(CFS, -CLIP, CLIP), extent=[W, E, S, N], cmap=cmap,
               norm=norm, origin="upper", alpha=0.78, interpolation="bilinear",
               zorder=2)

ax.contour(GX, GY, np.clip(CFS, -CLIP, CLIP), levels=[0.0], colors="0.15",
           linewidths=0.6, zorder=2.5)

ax.contour(np.linspace(W, E, z.shape[1]), np.linspace(N, S, z.shape[0]),
           lake.astype(float), levels=[0.5], colors="#2b5d8a",
           linewidths=0.9, zorder=3.3)
ax.text(42.92, 38.64, "Lake\nVan", ha="center", va="center", zorder=6,
        fontsize=11, style="italic", color="white", fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2.4, foreground="#173d5c")])

ax.add_patch(Polygon(poly, closed=True, facecolor="none", edgecolor="black",
                     lw=1.2, ls=(0, (4, 2)), zorder=5))
tr = np.array(trace)
ax.plot(tr[:, 0], tr[:, 1], color="black", lw=2.2, zorder=5.2,
        solid_capstyle="round")
nteeth = 7
for tt in np.linspace(0.08, 0.92, nteeth):
    px, py = tr[0] + tt * (tr[1] - tr[0])
    ax.add_patch(Polygon([(px, py),
                          (px + 0.028 * hvec[0] - 0.016 * (tr[1, 0] - tr[0, 0]) / L,
                           py + 0.028 * hvec[1]),
                          (px + 0.028 * hvec[0] + 0.016 * (tr[1, 0] - tr[0, 0]) / L,
                           py + 0.028 * hvec[1])],
                         closed=True, facecolor="black", zorder=5.2))

if eq2011.size:
    ax.scatter(eq2011[:, 0], eq2011[:, 1], s=8, marker="o", facecolor="0.25",
               edgecolor="none", alpha=0.45, zorder=5.5,
               label="2011 aftershocks ($M\\geq4$)")
ax.scatter([EPI_LON], [EPI_LAT], s=360, marker="*", facecolor="#ffe000",
           edgecolor="black", linewidth=1.3, zorder=8,
           label="2011 Van-Erci\u015f mainshock ($M_w$ 7.1)")
ax.scatter([VE_LON], [VE_LAT], s=170, marker="^", facecolor="#7d26cd",
           edgecolor="white", linewidth=1.1, zorder=8,
           label="Triggered Van-Edremit event ($M_w$ 5.6)")

for name, (lon, lat, ha) in TOWNS.items():
    ax.plot(lon, lat, "s", ms=6.5, mfc="white", mec="black", mew=1.3, zorder=7)
    off = 0.03 if ha == "left" else -0.03
    ax.text(lon + off, lat, name.replace("Ercis", "Erci\u015f"), fontsize=9,
            ha=ha, va="center", zorder=7, fontweight="bold",
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
ax.plot([x0, x0 + dlon], [y0, y0], color="black", lw=3.2, solid_capstyle="butt",
        zorder=9)
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
           mew=1.2, ms=15, label="2011 Van-Erci\u015f mainshock ($M_w$ 7.1)"),
    Line2D([0], [0], marker="^", color="none", mfc="#7d26cd", mec="white",
           mew=1.0, ms=11, label="Triggered Van-Edremit event ($M_w$ 5.6)"),
    Line2D([0], [0], marker="o", color="none", mfc="0.25", mec="none", ms=6,
           alpha=0.6, label="2011 aftershocks ($M\\geq4$)"),
    Line2D([0], [0], color="black", lw=2.2, label="2011 rupture (surface trace)"),
    Line2D([0], [0], color="black", lw=1.2, ls=(0, (4, 2)),
           label="Rupture-plane footprint"),
    Line2D([0], [0], marker="s", color="none", mfc="white", mec="black",
           mew=1.3, ms=6.5, label="Town"),
]
leg = ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.92,
                edgecolor="0.5", borderpad=0.6, handletextpad=0.6)
leg.set_zorder(10)

ax.set_title("Cascade\u2009III: Coulomb stress transfer from the 2011 "
             "$M_w$\u20097.1 Van earthquake\n$\\Delta$CFS on regional thrust "
             "faults, with the triggered Van\u2013Edremit event", fontsize=11,
             fontweight="bold", pad=8)

fig.subplots_adjust(left=0.075, right=0.865, top=0.895, bottom=0.180)
fig.canvas.draw()
pos = ax.get_position()
cax = fig.add_axes([pos.x1 + 0.018, pos.y0, 0.020, pos.height])
cb = fig.colorbar(SM(norm=norm, cmap=cmap), cax=cax, extend="both")
cb.set_label("$\\Delta$CFS on thrust receivers (MPa)", fontsize=9)
cb.set_ticks(np.arange(-CLIP, CLIP + 1e-9, 0.01))
cb.ax.yaxis.set_minor_locator(AutoMinorLocator(5))
cb.ax.tick_params(which="major", labelsize=8, length=4)
cb.ax.tick_params(which="minor", length=2.5)

CRED_FS = 8
segments = [
    "Depiction: Coulomb failure-stress change ($\\Delta$CFS, $\\mu'$=0.4) at "
    "7 km depth from the 2011 Van $M_w$\\,7.1 rupture, resolved on regional "
    "thrust receivers; 2011 mainshock, aftershocks ($M\\geq4$) and the "
    "triggered Van-Edremit $M_w$\\,5.6 event.",
    "Model: Okada (1992) dislocation (elastic half-space).  Data: GEBCO 2026 "
    "(15\u2033); EarthScope IEB (IRIS/USGS).  Software: Python (okada_wrapper, "
    "rasterio, matplotlib, numpy, scipy).  Source: authors.",
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

fig.savefig("fig12_cascade3_stress_transfer.png", dpi=200)
fig.savefig("fig12_cascade3_stress_transfer_hi.png", dpi=600)
fig.savefig("fig12_cascade3_stress_transfer.pdf")
print("saved | Van-Edremit dCFS = %+.1f kPa | CFS range [%.2f, %.2f] MPa | lines=%d"
      % (ve_val * 1000, np.nanmin(CFS), np.nanmax(CFS), len(lines)))
