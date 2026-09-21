import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, LineCollection
import matplotlib.patheffects as pe
import shapefile
from pyproj import CRS, Transformer

GEO = "geodata/geo4_2l/geo4_2l"
PRV = "geodata/prv4_2l/prv4_2l"
FLT = "geodata/flt4_2l/flt4_2l"
W, E, S, N = 42.2, 44.8, 38.0, 39.8
MIDLAT = 0.5 * (S + N)

src = CRS.from_wkt(open(GEO + ".prj").read())
to_ll = Transformer.from_crs(src, 4326, always_xy=True)
to_xy = Transformer.from_crs(4326, src, always_xy=True)

bx, by = [], []
for lo in np.linspace(W, E, 60):
    for la in (S, N):
        x, y = to_xy.transform(lo, la); bx.append(x); by.append(y)
for la in np.linspace(S, N, 60):
    for lo in (W, E):
        x, y = to_xy.transform(lo, la); bx.append(x); by.append(y)
NB = (min(bx), min(by), max(bx), max(by))
def hits(bb): return not (bb[2] < NB[0] or bb[0] > NB[2] or bb[3] < NB[1] or bb[1] > NB[3])

def rings(shape):
    pts = shape.points; parts = list(shape.parts) + [len(pts)]
    for i in range(len(parts) - 1):
        seg = pts[parts[i]:parts[i + 1]]
        if len(seg) >= 3:
            xs = [p[0] for p in seg]; ys = [p[1] for p in seg]
            lo, la = to_ll.transform(xs, ys)
            yield np.column_stack([lo, la])

COLOR = {
 'oth':'#ECECEC','PzpCmm':'#C7A9C9','PzpCm':'#B79AC0','Pzm':'#9FA8C7','Pzu':'#8FB0CB',
 'Pz':'#7FA9C4','Mzm':'#93B48C','P':'#E0A96B','T':'#9E86C0','TK':'#5FB6A6','K':'#8FCB8F',
 'CzMzi':'#E7739B','Pg':'#FDBE6F','N':'#FBEFA0','Czv':'#F98E6A','Qv':'#E23B34',
 'Q':'#FBFAC8','H2O':'#A9CFE7'}
LABEL = {
 'Q':'Quaternary','Qv':'Quaternary volcanic','N':'Neogene','Pg':'Paleogene',
 'Czv':'Cenozoic volcanic','CzMzi':'Cenozoic\u2013Mesozoic intrusive','K':'Cretaceous',
 'TK':'Triassic\u2013Cretaceous','T':'Triassic','P':'Permian','Mzm':'Mesozoic metamorphic',
 'Pz':'Paleozoic (undiff.)','Pzu':'Upper Paleozoic','Pzm':'Paleozoic metamorphic',
 'PzpCm':'Precambrian\u2013Paleozoic metaseds.','PzpCmm':'Precambrian\u2013Paleozoic metamorphic',
 'oth':'Undifferentiated','H2O':'Water (Lake Van)'}
DRAW = ['oth','PzpCmm','PzpCm','Pzm','Pzu','Pz','Mzm','P','T','TK','K','CzMzi','Pg','N','Czv','Qv','Q','H2O']
LEG = ['Q','Qv','N','Pg','Czv','CzMzi','K','TK','T','P','Mzm','Pz','Pzu','Pzm','PzpCm','PzpCmm','oth','H2O']

rg = shapefile.Reader(GEO + ".shp"); gi = [f[0] for f in rg.fields[1:]].index('GLG')
cats = {c: [] for c in DRAW}
for sr in rg.iterShapeRecords():
    if not hits(sr.shape.bbox):
        continue
    c = sr.record[gi]
    if c in cats:
        cats[c].extend(rings(sr.shape))

from matplotlib.path import Path as MplPath
rp = shapefile.Reader(PRV + ".shp"); pn = [f[0] for f in rp.fields[1:]].index('NAME')
prov = {}
for sr in rp.iterShapeRecords():
    if not hits(sr.shape.bbox):
        continue
    prov.setdefault(sr.record[pn], []).extend(rings(sr.shape))

PSTYLE = {
 'Araks':              (r'////',  '#0b3d5c'),
 'Lesser Caucasus':    (r'\\\\',  '#1b5e20'),
 'Zagros Thrust Zone': (r'xxxx',  '#5d4037'),
 'Zagros Fold Belt':   (r'....',  '#6d4c00'),
 'Kura Basin':         (r'++++',  '#4a148c'),
}

gx = np.linspace(W+0.02, E-0.02, 90); gy = np.linspace(S+0.02, N-0.02, 66)
GX, GY = np.meshgrid(gx, gy); gpts = np.column_stack([GX.ravel(), GY.ravel()])
prov_keep, prov_labels = [], []
for nm, rr in prov.items():
    inside = np.zeros(len(gpts), bool)
    for ring in rr:
        inside |= MplPath(ring).contains_points(gpts)
    if inside.mean() < 0.004:
        continue
    prov_keep.append(nm)
    cell = gpts[inside]; other = gpts[~inside]
    d2 = ((cell[:,None,0]-other[None,:,0])*np.cos(np.deg2rad(MIDLAT)))**2 \
       + (cell[:,None,1]-other[None,:,1])**2
    lx, ly = cell[d2.min(axis=1).argmax()]
    prov_labels.append((lx, ly, nm))
print("provinces kept:", prov_keep)

rf = shapefile.Reader(FLT + ".shp"); ti = [f[0] for f in rf.fields[1:]].index('TYPE')
faults = {}
for sr in rf.iterShapeRecords():
    if not hits(sr.shape.bbox):
        continue
    t = (sr.record[ti] or 'flt').strip() or 'flt'
    faults.setdefault(t, [])
    pts = sr.shape.points; parts = list(sr.shape.parts) + [len(pts)]
    for i in range(len(parts)-1):
        seg = pts[parts[i]:parts[i+1]]
        if len(seg) >= 2:
            xs=[p[0] for p in seg]; ys=[p[1] for p in seg]
            lo,la = to_ll.transform(xs,ys); faults[t].append(np.column_stack([lo,la]))

mpl.rcParams.update({'font.family':'DejaVu Sans','hatch.linewidth':0.5,'pdf.fonttype':42})
fig, ax = plt.subplots(figsize=(11.5, 8.8))

for c in DRAW:
    if not cats[c]:
        continue
    pc = PatchCollection([Polygon(a, closed=True) for a in cats[c]],
                         facecolor=COLOR[c], edgecolor='#5b5b5b', linewidths=0.12,
                         zorder=1)
    ax.add_collection(pc)

for nm in prov_keep:
    hatch, col = PSTYLE.get(nm, (r'////', '#333333'))
    polys = [Polygon(a, closed=True) for a in prov[nm]]
    ax.add_collection(PatchCollection(polys, facecolor='none', edgecolor=col,
                                      hatch=hatch, linewidths=0.0, alpha=0.9, zorder=3))
    ax.add_collection(PatchCollection(polys, facecolor='none', edgecolor=col,
                                      linewidths=1.8, linestyle=(0,(7,3)), zorder=3.6))

FCOL = {'flt':'#d7191c', 'th':'#7b3294'}
FLAB = {'flt':'Fault', 'th':'Thrust fault'}
for t, segs in faults.items():
    ax.add_collection(LineCollection(segs, colors='white', linewidths=3.0, alpha=0.7, zorder=4))
    ax.add_collection(LineCollection(segs, colors=FCOL.get(t,'#d7191c'), linewidths=1.9,
                                     zorder=4.5))

def _area_centroid(a):
    if not np.allclose(a[0], a[-1]): a = np.vstack([a, a[0]])
    x, y = a[:,0], a[:,1]; c = np.cos(np.deg2rad(y.mean())); xm = x*c
    cr = xm[:-1]*y[1:] - xm[1:]*y[:-1]; A = cr.sum()/2.0
    if abs(A) < 1e-12: return 0.0, (x.mean(), y.mean())
    cx = ((xm[:-1]+xm[1:])*cr).sum()/(6*A)/c
    cy = ((y[:-1]+y[1:])*cr).sum()/(6*A)
    return abs(A), (cx, cy)
def _rep_point(a, path):
    xmn,ymn = a.min(0); xmx,ymx = a.max(0)
    gx = np.linspace(xmn,xmx,9)[1:-1]; gy = np.linspace(ymn,ymx,9)[1:-1]
    if len(gx)==0 or len(gy)==0: return None
    P = np.array([(x,y) for y in gy for x in gx])
    ins = path.contains_points(P)
    if not ins.any(): return None
    cand = P[ins]
    d = np.min((cand[:,None,0]-a[None,:,0])**2 + (cand[:,None,1]-a[None,:,1])**2, axis=1)
    return cand[d.argmax()]
THR, NMAX, nlab = 0.0035, 7, 0
for code, rr in cats.items():
    if code in ('oth','H2O'):
        continue
    ranked = sorted(((_area_centroid(a), a) for a in rr), key=lambda t:-t[0][0])
    k = 0
    for (area, cen), a in ranked:
        if area < THR or k >= NMAX: break
        pth = MplPath(a); pt = cen if pth.contains_point(cen) else _rep_point(a, pth)
        if pt is None: continue
        px, py = pt
        if not (W+0.03 < px < E-0.03 and S+0.03 < py < N-0.03): continue
        ax.text(px, py, code, fontsize=7, fontweight='bold', color='#1a1a1a',
                ha='center', va='center', zorder=5.5,
                path_effects=[pe.withStroke(linewidth=1.8, foreground='white')])
        k += 1; nlab += 1
print("lithology labels:", nlab)

towns = [('Van',43.38,38.49,'left'),('Erci\u015f',43.36,39.03,'right'),
         ('\u00c7ald\u0131ran',43.90,39.14,'left'),('Muradiye',43.77,38.99,'left'),
         ('Edremit',43.24,38.42,'right'),('Ahlat',42.49,38.75,'right'),
         ('Tatvan',42.29,38.50,'left')]
halo=[pe.withStroke(linewidth=2.2, foreground='white')]
for nm,lo,la,ha in towns:
    if W<lo<E and S<la<N:
        ax.plot(lo,la,'s',ms=5,mfc='white',mec='black',mew=0.9,zorder=6)
        dx=0.03 if ha=='left' else -0.03
        ax.text(lo+dx,la,nm,fontsize=8.5,ha=ha,va='center',zorder=6,path_effects=halo)

ax.text(43.02,38.63,'Lake Van',fontsize=10,style='italic',color='#12405f',
        ha='center',va='center',zorder=6,path_effects=halo)

phalo=[pe.withStroke(linewidth=2.8,foreground='white')]
for lo,la,nm in prov_labels:
    col = PSTYLE.get(nm, (None,'#1a1a1a'))[1]
    if nm == 'Zagros Thrust Zone':
        ax.annotate(nm, xy=(lo,la), xytext=(43.05,38.22), color=col, fontsize=9,
                    fontweight='bold', ha='center', va='center', zorder=7,
                    path_effects=phalo,
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=1.8,
                                    path_effects=[pe.withStroke(linewidth=3.2,foreground='white')]))
    else:
        ax.text(lo,la,nm,fontsize=9,fontweight='bold',color=col,ha='center',
                va='center',zorder=7,path_effects=phalo)

ax.set_xlim(W,E); ax.set_ylim(S,N)
ax.set_aspect(1/np.cos(np.deg2rad(MIDLAT)))
ax.set_xticks(np.arange(42.5,44.6,0.5)); ax.set_yticks(np.arange(38.0,39.9,0.5))
ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,_:f"{v:.1f}\u00b0E"))
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,_:f"{v:.1f}\u00b0N"))
ax.tick_params(which='both',direction='out',length=4,labelsize=9,top=True,right=True)
ax.set_xticks(np.arange(42.2,44.81,0.1),minor=True)
ax.set_yticks(np.arange(38.0,39.81,0.1),minor=True)
ax.grid(True,which='major',color='0.6',lw=0.4,alpha=0.5)
for s in ax.spines.values(): s.set_linewidth(1.1)
ax.set_title("Geological map of the \u00c7ald\u0131ran\u2013Erci\u015f\u2013Van region, Eastern Anatolia",
             fontsize=12.5,pad=10)

kmperdeg = 111.320*np.cos(np.deg2rad(MIDLAT))
seg=50.0/kmperdeg
x0,y0=E-0.20-seg,S+0.13
ax.plot([x0,x0+seg],[y0,y0],'k-',lw=3,solid_capstyle='butt',zorder=7,
        path_effects=[pe.withStroke(linewidth=5,foreground='white')])
ax.text(x0+seg/2,y0+0.045,'50 km',ha='center',va='bottom',fontsize=8.5,zorder=7,
        path_effects=halo)

ax.annotate('N',xy=(E-0.16,N-0.10),xytext=(E-0.16,N-0.34),ha='center',va='center',
            fontsize=12,fontweight='bold',zorder=7,path_effects=halo,
            arrowprops=dict(arrowstyle='-|>',lw=2.2,color='black'))

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
handles=[Patch(facecolor=COLOR[c],edgecolor='#5b5b5b',lw=0.3,label=LABEL[c])
         for c in LEG if cats.get(c)]
handles.append(Patch(facecolor='none',edgecolor='none',label='Tectonic provinces:'))
for nm in prov_keep:
    h,col = PSTYLE.get(nm,(r'////','#333'))
    handles.append(Patch(facecolor='none',edgecolor=col,hatch=h,label=nm))
handles.append(Patch(facecolor='none',edgecolor='none',label=''))
handles.append(Line2D([0],[0],color='#d7191c',lw=2.2,label='Fault'))
handles.append(Line2D([0],[0],color='#7b3294',lw=2.2,label='Thrust fault'))
leg=ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(1.015,1.0),
              fontsize=8.3,frameon=True,handlelength=1.6,labelspacing=0.42,
              borderpad=0.7,title='Lithostratigraphic age',title_fontsize=9.5)
leg.get_frame().set_edgecolor('0.4'); leg.get_frame().set_linewidth(0.8)

ax.text(0.0,-0.065,
    "Data: geology, tectonic provinces and faults reprojected (WGS84) from USGS. "
    "Plotted with Matplotlib. Source: authors.",
    transform=ax.transAxes,fontsize=7.4,color='0.3',ha='left',va='top')

fig.savefig("outputs/fig02_geological_map.pdf",bbox_inches='tight')
fig.savefig("outputs/fig02_geological_map.png",dpi=600,bbox_inches='tight')
print("wrote fig02_geological_map.pdf/.png |",
      "geo cats:",sum(1 for c in DRAW if cats[c]),
      "| provinces:",len(prov_labels),"| fault types:",{t:len(v) for t,v in faults.items()})
