# Region grid around US-Var (Vaira Ranch)

All rasters share the same CRS (EPSG:32610, UTM zone 10N) and the same extent
(lon -121.25 to -120.8, lat 38.2 to 38.56),
upper-left corner x=652400 m, y=4270300 m; the 100 m grid is 403 x 409 pixels.
Coarser rasters (500 m MODIS, 1 km Daymet) cover the same box at their native resolution.
Band descriptions are stored in each GeoTIFF; int16 rasters carry a `scale_factor` tag.

| File | Content | Resolution |
|---|---|---|
| alphaearth_embedding_{year}_100m.tif | 64-band AlphaEarth annual embedding (int16 x 10000) | 100 m |
| alphaearth_similarity_to_2021_100m.tif | dot product of each year's 100 m mean embedding with that of 2021 | 100 m |
| landsat_ndvi_monthly_{year}_100m.tif | Landsat 8/9 monthly median NDVI, 30 m pixels averaged to 100 m (int16 x 10000) | 100 m |
| openet_ensemble_et_monthly_{year}_100m.tif | OpenET ensemble ET, mm/day (int16 x 100) | 100 m |
| elevation_100m.tif | NASADEM elevation, m | 100 m |
| daymet_monthly_2017_2024_1km.tif | monthly mean tmax, tmin (C), srad (W/m2), vp (Pa), dayl (s), prcp (mm/day) | 1 km |
| modis_vegetation_monthly_2017_2024_500m.tif | MOD13A1 NDVI/EVI and MCD15A3H LAI/fPAR monthly means | 500 m |
| mod16_et_monthly_2017_2024_500m.tif | MOD16A2GF ET, mm/day | 500 m |
| mod17_gpp_monthly_2017_2024_500m.tif | MOD17A2HGF GPP, gC/m2/day | 500 m |
| modis_landcover_2021_500m.tif | MCD12Q1 IGBP class | 500 m |

Sources: Google / Google DeepMind AlphaEarth Foundations (CC-BY 4.0); Daymet V4 (ORNL DAAC); Landsat (USGS);
MODIS (NASA LP DAAC); OpenET ensemble v2.0; NASADEM (NASA JPL).
