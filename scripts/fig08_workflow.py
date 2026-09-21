import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

C = {
    "I":   "#1f77b4",
    "II":  "#ff7f0e",
    "III": "#2ca02c",
    "IV":  "#d62728",
}
SHARE_FILL = "#ececec"
SHARE_EDGE = "#4d4d4d"
INK = "#1a1a1a"

def tint(hex_color, frac=0.13):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r = round(r + (255 - r) * (1 - frac))
    g = round(g + (255 - g) * (1 - frac))
    b = round(b + (255 - b) * (1 - frac))
    return f"#{r:02x}{g:02x}{b:02x}"

fig = plt.figure(figsize=(9.6, 6.9))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

GUT = 13.0
X0, X1 = 14.0, 99.3
NLANE = 4
GAP = 2.0
LW = (X1 - X0 - (NLANE - 1) * GAP) / NLANE
lane_left = [X0 + i * (LW + GAP) for i in range(NLANE)]
lane_cx = [xl + LW / 2 for xl in lane_left]
order = ["I", "II", "III", "IV"]

def box(x, y, w, h, fc, ec, lw=1.3, rad=0.9, z=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={rad}",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z,
                       mutation_aspect=0.62)
    ax.add_patch(p)
    return p

def txt(x, y, s, size, weight="normal", color=INK, style="normal", ha="center", va="center", z=4):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, style=style,
            ha=ha, va=va, zorder=z, linespacing=1.28)

def arrow(x0, y0, x1, y1, color=SHARE_EDGE, lw=1.5, z=1, rad=0.0):
    a = FancyArrowPatch((x0, y0), (x1, y1),
                        arrowstyle="-|>", mutation_scale=11,
                        lw=lw, color=color, zorder=z, shrinkA=0, shrinkB=0,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)

b_data   = (85.0, 96.0)
b_prep   = (72.0, 81.0)
hdr_y    = 67.6
b_feat   = (52.0, 65.0)
b_class  = (35.0, 48.0)
b_valid  = (18.0, 31.0)
b_map    = (3.5, 13.5)

def cy(b):
    return (b[0] + b[1]) / 2

stage = [
    (cy(b_data),  "Data\nacquisition", "open access"),
    (cy(b_prep),  "Pre-\nprocessing", None),
    (cy(b_feat),  "Feature\nengineering", None),
    (cy(b_class), "ML\nclassification", "RF · GBM · SVM"),
    (cy(b_valid), "Validation", "Acc · Prec\nRecall · F1 · AUC"),
    (cy(b_map),   "Cartographic\nmapping", "GMT · PyGMT"),
]
for yc, name, note in stage:
    txt(GUT/2, yc + (1.4 if note else 0), name, 8.4, weight="bold", color="#111111")
    if note:
        txt(GUT/2, yc - 3.3, note, 6.3, color="#555555", style="italic")

ax.add_line(Line2D([GUT, GUT], [1.5, 97.5], color="#cccccc", lw=0.8, zorder=0))

box(X0, b_data[0], X1 - X0, b_data[1]-b_data[0], SHARE_FILL, SHARE_EDGE, lw=1.4)
txt((X0+X1)/2, b_data[1]-2.0, "Open-access datasets", 9.2, weight="bold")
data_body = ("SRTM/ASTER DEM  ·  GEBCO & Lake Van bathymetry  ·  satellite imagery  ·  geological maps  ·  "
             "active-fault database\n"
             "earthquake catalogues (1976, 2011 sequences)  ·  field landslide & liquefaction inventories  ·  "
             "seismic-reflection profiles\n"
             "& ICDP PaleoVan cores  ·  regional coseismic-slip / stress models")
txt((X0+X1)/2, b_data[0]+ (b_data[1]-b_data[0])*0.40, data_body, 7.1)

arrow((X0+X1)/2, b_data[0], (X0+X1)/2, b_prep[1])

box(X0, b_prep[0], X1 - X0, b_prep[1]-b_prep[0], SHARE_FILL, SHARE_EDGE, lw=1.4)
txt((X0+X1)/2, b_prep[1]-2.0, "Preprocessing & conditioning", 9.2, weight="bold")
prep_body = ("reprojection to a common CRS  ·  DEM void-filling & hydrological conditioning  ·  "
             "fault-line digitisation & topology cleaning\n"
             "earthquake-catalogue declustering & magnitude-of-completeness assessment  ·  raster co-registration")
txt((X0+X1)/2, b_prep[0]+(b_prep[1]-b_prep[0])*0.36, prep_body, 7.1)

for cx in lane_cx:
    arrow((X0+X1)/2, b_prep[0], cx, b_feat[1], rad=0.0 if abs(cx-(X0+X1)/2) < 8 else 0.06*( -1 if cx < (X0+X1)/2 else 1))

hdr = {
    "I":   ("I", "Coseismic slope failure", "1976 Çaldıran  $M_w$7.3"),
    "II":  ("II", "Liquefaction & landsliding", "2011 Van–Erciş  $M_w$7.1"),
    "III": ("III", "Stress-triggered seismicity", "2011 Van–Edremit  $M_w$5.6"),
    "IV":  ("IV", "Sublacustrine mass wasting", "multi-centennial record"),
}
for cx, k in zip(lane_cx, order):
    box(cx-LW/2, hdr_y-2.2, LW, 4.4, C[k], C[k], lw=0, rad=0.7, z=3)
    rn, title, sub = hdr[k]
    txt(cx-LW/2+2.6, hdr_y, rn, 10.5, weight="bold", color="white", ha="center", z=5)
    txt(cx-LW/2+5.8, hdr_y+0.9, title, 6.7, weight="bold", color="white", ha="left", va="center", z=5)
    txt(cx-LW/2+5.8, hdr_y-1.3, sub, 6.2, color="white", ha="left", va="center", z=5)

feat = {
    "I":   "Slope · aspect · curvature\nlithology · distance-to-fault\nPGA / shaking proxy",
    "II":  "Groundwater depth\nsediment type · slope\nsaturation proxy",
    "III": "Coseismic-slip model\nreceiver-fault geometry\n$\\Delta$CFS  (Eq. 5)",
    "IV":  "Bathymetric slope\nsediment thickness\nfault proximity",
}
for cx, xl, k in zip(lane_cx, lane_left, order):
    box(xl, b_feat[0], LW, b_feat[1]-b_feat[0], tint(C[k]), C[k], lw=1.3)
    txt(cx, b_feat[1]-1.8, "Predictor features", 7.2, weight="bold", color=C[k])
    txt(cx, b_feat[0]+(b_feat[1]-b_feat[0])*0.40, feat[k], 6.9)
    arrow(cx, b_feat[0], cx, b_class[1], color=C[k])

clf = {
    "I":   "Rockfall / landslide\nsusceptibility",
    "II":  "Liquefaction /\nlateral-spread\nsusceptibility",
    "III": "Secondary-seismicity\nprobability",
    "IV":  "Mass-wasting\nsusceptibility",
}
for cx, xl, k in zip(lane_cx, lane_left, order):
    box(xl, b_class[0], LW, b_class[1]-b_class[0], tint(C[k]), C[k], lw=1.3)
    txt(cx, b_class[1]-1.8, "Supervised classifier", 7.2, weight="bold", color=C[k])
    txt(cx, b_class[0]+(b_class[1]-b_class[0])*0.42, clf[k], 7.0)
    arrow(cx, b_class[0], cx, b_valid[1], color=C[k])

val = {
    "I":   "1976 Çaldıran rockfall\n/ landslide inventory",
    "II":  "2011 liquefaction &\nlandslide inventory",
    "III": "2011 Van–Edremit\naftershock location",
    "IV":  "mapped slides /\nturbidites (cores,\nseismic profiles)",
}
for cx, xl, k in zip(lane_cx, lane_left, order):
    box(xl, b_valid[0], LW, b_valid[1]-b_valid[0], tint(C[k]), C[k], lw=1.3)
    txt(cx, b_valid[1]-1.8, "Ground-truth check", 7.2, weight="bold", color=C[k])
    txt(cx, b_valid[0]+(b_valid[1]-b_valid[0])*0.40, val[k], 6.9)

    arrow(cx, b_valid[0], (X0+X1)/2, b_map[1],
          color=C[k], rad=0.0 if abs(cx-(X0+X1)/2) < 8 else 0.05*(-1 if cx < (X0+X1)/2 else 1))

box(X0, b_map[0], X1 - X0, b_map[1]-b_map[0], SHARE_FILL, SHARE_EDGE, lw=1.4)
txt((X0+X1)/2, b_map[1]-2.0, "GMT / PyGMT hazard mapping", 9.2, weight="bold")
map_body = ("four cascade-susceptibility maps  +  integrated multi-hazard synthesis map\n"
            "for the Çaldıran–Erciş–Van fault system and the Lake Van basin")
txt((X0+X1)/2, b_map[0]+(b_map[1]-b_map[0])*0.36, map_body, 7.2)

fig.savefig("fig08_workflow.png", dpi=200, bbox_inches="tight", pad_inches=0.04)
fig.savefig("fig08_workflow_hi.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
fig.savefig("fig08_workflow.pdf", bbox_inches="tight", pad_inches=0.04)
print("saved")
