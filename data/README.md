# Input data (open-access — not committed to the repository)

Place the following open-access files here. They are excluded from version
control by `.gitignore`; download them from the sources below.

## Rasters and catalogues (this `data/` folder)

| File | Description | Source |
|---|---|---|
| `gebco_2026_n39_4_s38_0_w42_3_e44_3_geotiff.tif` | GEBCO 2026 topo–bathymetry grid clipped to the study window | GEBCO (gebco.net) |
| `IEB_Turkey_5000_events.csv` | Instrumental earthquake catalogue | KOERI / Boğaziçi University (koeri.boun.edu.tr) |
| `TP_Arabian.txt`, `TP_Eurasian.txt` | Plate-boundary coordinate traces | published plate-boundary model (e.g. Bird, 2003) |
| `region_relief.tif`, `inset_relief.tif` | Derived relief rasters for the location insets | generated from the GEBCO grid |

## Vector layers (`../geodata/`)

Shapefile sets (`.shp` + `.dbf` + `.shx` + `.prj`), referenced by stem name:

| Stem | Description |
|---|---|
| `geodata/flt4_2l/flt4_2l` | Active faults |
| `geodata/geo4_2l/geo4_2l` | Geological units |
| `geodata/prv4_2l/prv4_2l` | Administrative / provincial boundaries |

Note: `fig01_regional_setting.py` also searches the current directory and a
`grids/` folder for the bare-named rasters, so paths are flexible.
