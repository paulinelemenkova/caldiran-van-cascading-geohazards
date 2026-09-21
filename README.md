# Cascading geohazards of the Çaldıran–Van fault–lake system (Eastern Anatolia)

Reproducible figure-generation code for a study of crustal deformation and
earthquake-triggered **cascading geohazards** across the Çaldıran, Erçiş and
Van fault systems of Eastern Anatolia, and their long-term expression in the
late Quaternary archive of Lake Van. The analysis combines open-access
geological, seismic-catalogue, remote-sensing, space-geodetic and lacustrine
data with reproducible terrain and geospatial mapping.

**Authors:** Polina Lemenkova, Abdullah Can Zülfikar
(Institute of Earthquake Engineering and Disaster Management,
Istanbul Technical University).

> Code accompanying a manuscript currently under review. This repository
> contains the geospatial figure-reproduction scripts; the machine-learning
> classification/sensitivity engine and its derived result files are not part
> of this public release.

## Repository layout

```
scripts/     figure-generation scripts (fig01–fig13)
data/        user-supplied open-access rasters/catalogues (see data/README.md)
geodata/     user-supplied vector layers (faults, geology, boundaries)
figures/     reference outputs produced by the scripts
```

## What each script produces

| Script | Figure |
|---|---|
| `fig01_regional_setting.py` | Regional tectonic setting and location inset |
| `fig02_geological_map.py` | Geological map of the study area |
| `fig03_dem_relief.py` | DEM shaded relief with seismicity |
| `fig04_active_faults.py` | Active-fault map |
| `fig05_cross_section.py` | Crustal cross-section |
| `fig06_seismicity_map.py` | Instrumental seismicity map |
| `fig07_gutenberg_richter.py` | Gutenberg–Richter frequency–magnitude statistics |
| `fig08_workflow.py` | Methodological workflow diagram |
| `fig09_feature_importance.py` | Predictor correlation matrix and importance |
| `fig10_cascade1_rockfall_landslide.py` | Cascade I susceptibility (rockfall/landslide) |
| `fig11_cascade2_liquefaction.py` | Cascade II susceptibility (liquefaction) |
| `fig12_cascade3_stress_transfer.py` | Cascade III (Coulomb stress transfer) |
| `fig13_cascade4_lake_van.py` | Cascade IV (sublacustrine mass wasting, Lake Van) |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then place the open-access input data as described in `data/README.md`
(rasters/catalogues in `data/`, vector layers in `geodata/`).

## Running

Each script is standalone and writes its figure (PNG/PDF) to the working
directory:

```bash
python3 scripts/fig03_dem_relief.py
```

## Data availability

All inputs are open-access (GEBCO bathymetry/topography, the KOERI/Boğaziçi
instrumental earthquake catalogue, published active-fault and geological
vector layers, and plate-boundary coordinates). See `data/README.md` for
sources and expected filenames. Large data files are intentionally excluded
from version control (see `.gitignore`).

## License

Code released under the MIT License (see `LICENSE`).

## Citation

If you use this code, please cite it using the metadata in `CITATION.cff`
(GitHub renders a "Cite this repository" button from it).
