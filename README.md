# CEE 491/591 Process Hydrology Tutorials

Hands-on Python tutorials for the CEE 491/591 Process Hydrology course at The University of Alabama. Each notebook
introduces the physics first, then applies it to real observations, with worked code, figures, exercises, and references.
The tutorials run on Google Colab (click a badge) or locally after cloning the repository.

Most tutorials use the daily record of the AmeriFlux site **US-Var (Vaira Ranch, California, 2000-2021)** included in
`sample_data/`: fluxes, meteorology, LAI, fPAR, GPP, and soil moisture, plus Daymet precipitation. The streamflow tutorial
uses USGS daily discharge for the Cahaba River at Trussville, Alabama (2000-2023). All data files are in `sample_data/`;
the notebooks download them automatically when run on Colab.

## Process hydrology

| Tutorial | What you will do | Colab |
|---|---|---|
| [The energy imbalance problem and its corrections](tutorials/process_hydrology/01_Energy_Imbalance_Problem_and_Corrections.ipynb) | Surface energy balance, closure across the global flux network and at one site, five closure corrections (Bowen ratio, OFC, MDEBR, PULSE, FLARE) applied to half-hourly data, and their effect on ET | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praghav444/CEE491-591-Process-Hydrology-Tutorials/blob/main/tutorials/process_hydrology/01_Energy_Imbalance_Problem_and_Corrections.ipynb) |
| [PET methods and the Priestley-Taylor coefficient](tutorials/process_hydrology/02_PET_Methods_and_Priestley_Taylor_Alpha.ipynb) | Priestley-Taylor, FAO-56 reference ET, Hargreaves; calibrate alpha against measured ET | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praghav444/CEE491-591-Process-Hydrology-Tutorials/blob/main/tutorials/process_hydrology/02_PET_Methods_and_Priestley_Taylor_Alpha.ipynb) |
| [Penman-Monteith-based ET modeling](Penman-Monteith-based-ET-Modeling-Tutorial.ipynb) | The full Penman-Monteith equation: aerodynamic and surface conductance, stability corrections, surface resistance from LAI, applied to flux-tower data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praghav444/CEE491-591-Process-Hydrology-Tutorials/blob/main/Penman-Monteith-based-ET-Modeling-Tutorial.ipynb) |
| [ET partitioning with water-use efficiency](tutorials/process_hydrology/03_ET_Partitioning_Water_Use_Efficiency.ipynb) | Underlying WUE, quantile regression for the potential WUE, transpiration fraction T/ET | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praghav444/CEE491-591-Process-Hydrology-Tutorials/blob/main/tutorials/process_hydrology/03_ET_Partitioning_Water_Use_Efficiency.ipynb) |
| [Infiltration and soil water retention](tutorials/process_hydrology/04_Infiltration_and_Soil_Water_Retention.ipynb) | Green-Ampt with ponding time, van Genuchten retention and conductivity, plant-available water | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praghav444/CEE491-591-Process-Hydrology-Tutorials/blob/main/tutorials/process_hydrology/04_Infiltration_and_Soil_Water_Retention.ipynb) |
| [Root-zone soil moisture bucket model](tutorials/process_hydrology/05_Root_Zone_Soil_Moisture_Bucket_Model.ipynb) | Build a daily bucket model, calibrate it to measured soil moisture, validate on later years | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praghav444/CEE491-591-Process-Hydrology-Tutorials/blob/main/tutorials/process_hydrology/05_Root_Zone_Soil_Moisture_Bucket_Model.ipynb) |
| [Streamflow, baseflow, and flood frequency](tutorials/process_hydrology/06_Streamflow_Baseflow_and_Flood_Frequency.ipynb) | Hydrographs, flow duration curve, Lyne-Hollick baseflow filter, runoff ratio, Gumbel and log-normal floods | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praghav444/CEE491-591-Process-Hydrology-Tutorials/blob/main/tutorials/process_hydrology/06_Streamflow_Baseflow_and_Flood_Frequency.ipynb) |
| [Drought indices: SPI, SPEI, ESR](tutorials/process_hydrology/07_Drought_Indices_SPI_SPEI_ESR.ipynb) | Standardized indices from precipitation and P-PET, evaporative stress ratio, California droughts 2007-2016 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praghav444/CEE491-591-Process-Hydrology-Tutorials/blob/main/tutorials/process_hydrology/07_Drought_Indices_SPI_SPEI_ESR.ipynb) |
| [Flash drought detection using the evaporative stress ratio](Flash_Drought_Detection_using_ESR.ipynb) | Identify rapid-onset droughts from ET/PET with the Christian et al. (2019) criteria | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praghav444/CEE491-591-Process-Hydrology-Tutorials/blob/main/Flash_Drought_Detection_using_ESR.ipynb) |

## Geo-AI: machine learning for hydrology

| Tutorial | What you will do | Colab |
|---|---|---|
| [Machine learning for ET: a random forest baseline](tutorials/geo_ai/08_Machine_Learning_for_ET_Random_Forest.ipynb) | Time-based splits, NSE, permutation importance, partial dependence, physical limits and extrapolation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praghav444/CEE491-591-Process-Hydrology-Tutorials/blob/main/tutorials/geo_ai/08_Machine_Learning_for_ET_Random_Forest.ipynb) |
| [Physics-guided hybrid model: a neural network inside Penman-Monteith](tutorials/geo_ai/09_Physics_Guided_Hybrid_Penman_Monteith_Neural_Network.ipynb) | Differentiable Penman-Monteith in PyTorch, learned surface conductance, comparison with a pure network | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praghav444/CEE491-591-Process-Hydrology-Tutorials/blob/main/tutorials/geo_ai/09_Physics_Guided_Hybrid_Penman_Monteith_Neural_Network.ipynb) |
| [Gap-filling flux data with machine learning](tutorials/geo_ai/10_Gap_Filling_Flux_Data_with_Machine_Learning.ipynb) | Artificial short and long gaps, MDS-style lookup versus random forest, effect on annual ET | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praghav444/CEE491-591-Process-Hydrology-Tutorials/blob/main/tutorials/geo_ai/10_Gap_Filling_Flux_Data_with_Machine_Learning.ipynb) |
| [Satellite data with STAC: Sentinel-2 NDVI](tutorials/geo_ai/11_Satellite_Data_with_STAC_Sentinel2_NDVI.ipynb) | Search a STAC catalog, read windows from cloud-optimized GeoTIFFs, NDVI time series versus tower LAI | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praghav444/CEE491-591-Process-Hydrology-Tutorials/blob/main/tutorials/geo_ai/11_Satellite_Data_with_STAC_Sentinel2_NDVI.ipynb) |
| [Geo foundation models for ET and GPP: AlphaEarth embeddings](tutorials/geo_ai/12_Geo_Foundation_Models_AlphaEarth_ET_GPP.ipynb) | What an embedding is, analog towers and linear probes, traditional versus embedding-based flux models at 200+ towers with leave-site-out validation and learning curves, monthly ET and GPP maps at 100 m around Vaira Ranch against OpenET, MOD16, MOD17 and the withheld towers, change detection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praghav444/CEE491-591-Process-Hydrology-Tutorials/blob/main/tutorials/geo_ai/12_Geo_Foundation_Models_AlphaEarth_ET_GPP.ipynb) |

## Running locally

```
git clone https://github.com/praghav444/CEE491-591-Process-Hydrology-Tutorials.git
cd CEE491-591-Process-Hydrology-Tutorials
pip install -r requirements.txt
jupyter lab
```

All tutorials except the two below need only `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, and `statsmodels`.
The physics-guided hybrid model needs `torch` (CPU is enough). The satellite tutorial needs `pystac-client` and `rasterio` and an internet connection.
The foundation-model tutorial needs `rasterio` and downloads about 60 MB of prepared data on Colab; no Earth Engine account is required.

## Data

| File | Content | Source |
|---|---|---|
| `sample_data/daily_data_UA-Var_2000_2021.csv` | Daily fluxes (LE, H, G), net radiation, meteorology, LAI, fPAR, GPP, soil moisture, PET at US-Var | AmeriFlux / FLUXNET2015 processing |
| `sample_data/daymet_US-Var_2000_2021.csv` | Daily precipitation, Tmax, Tmin, radiation, vapor pressure, day length at US-Var | Daymet V4 single-pixel service |
| `sample_data/usgs_02423130_Cahaba_daily_discharge_2000_2023.csv` | Daily mean discharge, Cahaba River at Trussville, AL (19.7 sq mi) | USGS Water Services |
| `sample_data/daymet_Cahaba_02423130_2000_2023.csv` | Daily weather at the Cahaba gauge | Daymet V4 |
| `sample_data/fitted_alpha_PT_params.csv` | Fitted Priestley-Taylor coefficients used in the flash drought tutorial | This repository |
| `Sample_data.csv` | One year of daily flux-tower data used in the Penman-Monteith tutorial | This repository |
| `sample_data/alphaearth/sites.csv` | 289 AmeriFlux towers in North America with data in 2017-2024: coordinates, IGBP class, Koppen climate, elevation, MODIS land cover | AmeriFlux FLUXNET (ONEFlux) product, NASADEM, MCD12Q1 |
| `sample_data/alphaearth/site_year_embeddings.csv` | 64-D AlphaEarth annual embedding per tower and year (100 m footprint mean) | Google / Google DeepMind Satellite Embedding V1 (CC-BY 4.0) via Earth Engine |
| `sample_data/alphaearth/site_month_data.csv` | Monthly tower ET (uncorrected and energy-balance corrected), GPP, meteorology, plus Daymet, Landsat 8/9 NDVI and MODIS NDVI/EVI/LAI/fPAR at each tower | AmeriFlux FLUXNET, Daymet V4, USGS Landsat, MODIS |
| `sample_data/alphaearth/region/` | 40 km x 40 km rasters around Vaira Ranch (UTM 10N): AlphaEarth embeddings 2021 and 2023 at 100 m, year-to-year similarity, Landsat NDVI, OpenET ensemble ET, MOD16 ET, MOD17 GPP, Daymet, MODIS vegetation and land cover, elevation | see `region/README.md` |
| `sample_data/alphaearth/prep_alphaearth_data.py` | The Earth Engine script that produced the files above (needs an Earth Engine credential and the FLUXNET zips) | This repository |

Daymet citation: Thornton et al. (2022), Daymet: Daily Surface Weather Data on a 1-km Grid for North America, Version 4 R1, ORNL DAAC.
AlphaEarth citation: Brown, C. F., et al. (2025), AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data, arXiv:2507.22291; dataset `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, Google and Google DeepMind, CC-BY 4.0.
OpenET citation: Melton, F. S., et al. (2022), OpenET: Filling a critical data gap in water management for the western United States, JAWRA 58, 971-994. AmeriFlux FLUXNET data are shared under CC-BY 4.0; see each site's data policy for attribution.
USGS data are provisional and subject to revision.

## License

MIT License. If you use these tutorials in a course, a link back is appreciated.

Pushpendra Raghav, The University of Alabama. ppushpendra@ua.edu
