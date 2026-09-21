"""
One-time data preparation for Tutorial 12 (geo foundation models for ET and GPP).

Produces the files in sample_data/alphaearth/ so that the notebook runs WITHOUT an
Earth Engine account. Needs: earthengine-api, numpy, pandas, rasterio, pyproj, and
a Google Earth Engine credential (service-account key file or an interactive login).

Usage (from the repository root):
    python sample_data/alphaearth/prep_alphaearth_data.py --key path/to/gee-key.json \
        --flux-dir path/to/FLUXNET/zips --metadata path/to/FINAL_SITE_METADATA.csv

Every output is written as soon as it is ready and skipped on a re-run, so the
script can be restarted safely.

Outputs
-------
sites.csv                        one row per tower: id, lat, lon, IGBP, Koppen, elevation, MODIS land cover
site_year_embeddings.csv         64-D AlphaEarth embedding (100 m footprint mean) per site-year, 2017-2024
site_month_data.csv              monthly tower ET/GPP + Daymet + Landsat NDVI + MODIS NDVI/EVI/LAI/fPAR per site-month
region/*.tif                     40 km x 40 km grid around US-Var in EPSG:32610 (see region/README.md)
"""
import argparse, json, os, re, time, zipfile, io, datetime, calendar
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
YEARS = list(range(2017, 2025))           # AlphaEarth annual embeddings exist for 2017-2024
MAP_YEARS = [2021, 2023]                  # years with the full 64-D grid, 100 m Landsat NDVI and OpenET (drought / wet)
FOOTPRINT_RADIUS_M = 100                  # tower "footprint" used to average 10 m embeddings and 30 m NDVI
REGION_BBOX = (-121.25, 38.20, -120.80, 38.56)   # lon_min, lat_min, lon_max, lat_max around US-Var
REGION_CRS = "EPSG:32610"                 # UTM zone 10N
GRID_RES = 100                            # m, grid for embeddings, Landsat NDVI, OpenET, elevation
EMB_SCALE = 10000                         # embeddings stored as int16 = round(value * EMB_SCALE)
US_VAR = (-120.9508, 38.4133)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
REG = os.path.join(OUT, "region")
os.makedirs(REG, exist_ok=True)


def log(*a):
    print(datetime.datetime.now().strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------------------
# 1. Sites: metadata x available FLUXNET zips (North America, data in 2017 or later)
# --------------------------------------------------------------------------------------
def build_site_table(flux_dir, metadata_csv):
    meta = pd.read_csv(metadata_csv)
    rows = []
    for z in os.listdir(flux_dir):
        m = re.match(r"^[A-Za-z]+_([A-Z]{2}-[A-Za-z0-9]{3})_FLUXNET_(\d{4})-(\d{4})_v[\d.]+_r\d+\.zip$", z)
        if m:
            rows.append((m.group(1), int(m.group(2)), int(m.group(3)), z))
    zdf = pd.DataFrame(rows, columns=["site_id", "zip_start", "zip_end", "zip"]).drop_duplicates("site_id", keep="last")
    df = meta.merge(zdf, on="site_id")
    df = df[df.site_id.str[:2].isin(["US", "CA", "MX"]) & (df.zip_end >= YEARS[0])].copy()
    df = df[["site_id", "latitude", "longitude", "land_cover", "Koppen", "Elev_gtopo30", "zip_start", "zip_end", "zip"]]
    df = df.rename(columns={"latitude": "lat", "longitude": "lon", "land_cover": "igbp", "Koppen": "koppen", "Elev_gtopo30": "elev_gtopo30"})
    return df.sort_values("site_id").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# 2. Monthly tower data from the FLUXNET MM files
# --------------------------------------------------------------------------------------
MM_COLS = ["TIMESTAMP", "TA_F", "SW_IN_F", "LW_IN_F", "VPD_F", "PA_F", "P_F", "WS_F", "NETRAD", "G_F_MDS",
           "LE_F_MDS", "LE_F_MDS_QC", "LE_CORR", "H_F_MDS", "H_F_MDS_QC", "H_CORR",
           "NEE_VUT_REF", "NEE_VUT_REF_QC", "GPP_NT_VUT_REF", "GPP_DT_VUT_REF", "SWC_F_MDS_1", "TS_F_MDS_1"]


def read_flux_monthly(flux_dir, site_row):
    zpath = os.path.join(flux_dir, site_row.zip)
    with zipfile.ZipFile(zpath) as zf:
        names = [n for n in zf.namelist() if "FLUXMET_MM" in n and n.endswith(".csv")]
        if not names:
            return None
        with zf.open(names[0]) as f:
            d = pd.read_csv(io.BytesIO(f.read()), na_values=[-9999, -9999.0])
    keep = [c for c in MM_COLS if c in d.columns]
    d = d[keep].copy()
    for c in MM_COLS:
        if c not in d.columns:
            d[c] = np.nan
    d["year"] = d.TIMESTAMP // 100
    d["month"] = d.TIMESTAMP % 100
    d = d[(d.year >= YEARS[0]) & (d.year <= YEARS[-1])]
    # latent heat of vaporization (MJ/kg) and ET in mm/day (monthly mean of daily ET)
    lam = 2.501 - 0.002361 * d.TA_F
    d["ET_obs"] = d.LE_F_MDS * 0.0864 / lam
    d["ET_obs_corr"] = d.LE_CORR * 0.0864 / lam
    d["GPP_obs"] = d.GPP_NT_VUT_REF
    d["GPP_obs_dt"] = d.GPP_DT_VUT_REF
    d.insert(0, "site_id", site_row.site_id)
    return d.drop(columns=["TIMESTAMP"])


# --------------------------------------------------------------------------------------
# 3. Earth Engine helpers
# --------------------------------------------------------------------------------------
def ee_init(key):
    import ee
    if key and key.endswith(".json"):
        sa = json.load(open(key))["client_email"]
        ee.Initialize(ee.ServiceAccountCredentials(sa, key), opt_url="https://earthengine-highvolume.googleapis.com")
    else:
        ee.Initialize(project=key or None, opt_url="https://earthengine-highvolume.googleapis.com")
    return ee


def retry(fn, tries=4, wait=20):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa
            log(f"   retry {i + 1}/{tries}: {str(e)[:160]}")
            time.sleep(wait * (i + 1))
    raise RuntimeError("Earth Engine call failed repeatedly")


def month_range(ee, y, m):
    start = ee.Date.fromYMD(y, m, 1)
    return start, start.advance(1, "month")


def landsat_ndvi_monthly(ee, y, m, region=None):
    """Median cloud-masked NDVI from Landsat 8 + 9 Collection 2 L2 for one month."""
    start, end = month_range(ee, y, m)

    def prep(img):
        qa = img.select("QA_PIXEL")
        mask = (qa.bitwiseAnd(1 << 1).eq(0).And(qa.bitwiseAnd(1 << 2).eq(0))
                .And(qa.bitwiseAnd(1 << 3).eq(0)).And(qa.bitwiseAnd(1 << 4).eq(0)))
        sr = img.select(["SR_B4", "SR_B5"]).multiply(0.0000275).add(-0.2)
        ndvi = sr.normalizedDifference(["SR_B5", "SR_B4"]).rename("ndvi")
        return ndvi.updateMask(mask).updateMask(sr.select("SR_B4").gt(0))

    coll = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
            .filterDate(start, end).filter(ee.Filter.lt("CLOUD_COVER", 80)))
    if region is not None:
        coll = coll.filterBounds(region)
    coll = coll.map(prep)
    img = ee.Image(ee.Algorithms.If(coll.size().gt(0), coll.median(), ee.Image.constant(0).updateMask(0).toFloat()))
    return img.rename("ndvi").toFloat()


def daymet_monthly(ee, y, m):
    start, end = month_range(ee, y, m)
    ic = ee.ImageCollection("NASA/ORNL/DAYMET_V4").filterDate(start, end)
    mean = ic.select(["tmax", "tmin", "srad", "vp", "dayl"]).mean()
    prcp = ic.select("prcp").mean().rename("prcp")          # mean daily precipitation, mm/day
    return mean.addBands(prcp).toFloat()


def modis_monthly(ee, y, m):
    start, end = month_range(ee, y, m)
    vi = ee.ImageCollection("MODIS/061/MOD13A1").filterDate(start, end).select(["NDVI", "EVI"]).mean().multiply(0.0001)
    vi = vi.rename(["ndvi_modis", "evi_modis"])
    lai = ee.ImageCollection("MODIS/061/MCD15A3H").filterDate(start, end).select(["Lai", "Fpar"]).mean()
    lai = lai.multiply(ee.Image.constant([0.1, 0.01])).rename(["lai_modis", "fpar_modis"])
    return vi.addBands(lai).toFloat()


def modis_8day_to_monthly_rate(ee, coll_id, band, scale, y, m, valid_max):
    """Convert an 8-day MODIS total (MOD16 ET, MOD17 GPP) to a monthly mean daily rate.
    Each composite is turned into a daily rate (total / number of days it covers), then
    the composites starting in the month are averaged. Fill values (> valid_max) are masked."""
    start, end = month_range(ee, y, m)
    ic = ee.ImageCollection(coll_id).filterDate(start, end).select(band)

    def rate(img):
        doy = ee.Number(img.date().getRelative("day", "year")).add(1)
        leap = ee.Number(ee.Algorithms.If(img.date().get("year").mod(4).eq(0), 1, 0))
        ndays = ee.Number(ee.Algorithms.If(doy.gte(361), ee.Number(5).add(leap), 8))
        return img.updateMask(img.lte(valid_max)).multiply(scale).divide(ndays).toFloat().copyProperties(img, ["system:time_start"])

    return ic.map(rate).mean().toFloat()


def openet_monthly(ee, y):
    """OpenET ensemble monthly ET (mm/month) -> mm/day for the 12 months of a year."""
    ic = ee.ImageCollection("projects/openet/assets/ensemble/conus/gridmet/monthly/v2_0")
    bands = []
    for m in range(1, 13):
        start, end = month_range(ee, y, m)
        img = ic.filterDate(start, end).select("et_ensemble_mad").first()
        bands.append(ee.Image(img).toFloat().divide(calendar.monthrange(y, m)[1]).rename(f"{y}-{m:02d}"))
    return ee.Image.cat(bands).toFloat()


def embedding_year(ee, y):
    return ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate(f"{y}-01-01", f"{y + 1}-01-01").mosaic()


def mean_to_grid(ee, img, native_scale):
    """Make computePixels AVERAGE the native pixels inside each coarser output pixel instead of picking the nearest one."""
    return img.setDefaultProjection(REGION_CRS, None, native_scale).reduceResolution(ee.Reducer.mean(), True, 256)


# --------------------------------------------------------------------------------------
# 4. Site sampling
# --------------------------------------------------------------------------------------
def site_fc(ee, sites):
    feats = [ee.Feature(ee.Geometry.Point([r.lon, r.lat]).buffer(FOOTPRINT_RADIUS_M), {"site_id": r.site_id})
             for r in sites.itertuples()]
    return ee.FeatureCollection(feats)


def sample(ee, image, fc, scale, props):
    """Mean of `image` over each footprint polygon; returns DataFrame indexed by site_id."""
    def run():
        res = image.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=scale, tileScale=4).getInfo()
        rows = []
        for f in res["features"]:
            p = f["properties"]
            if len(props) == 1:                      # a single-band image reduces to a property called "mean"
                p = {props[0]: p.get("mean", np.nan), "site_id": p["site_id"]}
            rows.append({"site_id": p["site_id"], **{k: p.get(k, np.nan) for k in props}})
        return pd.DataFrame(rows).set_index("site_id")
    return retry(run)


def extract_sites(ee, sites):
    fc = site_fc(ee, sites)
    # static: elevation and MODIS land cover
    f_sites = os.path.join(OUT, "sites.csv")
    if not os.path.exists(f_sites):
        log("sites: elevation + MODIS land cover")
        elev = ee.Image("NASA/NASADEM_HGT/001").select("elevation").rename("elev_m")
        lc = ee.ImageCollection("MODIS/061/MCD12Q1").filterDate("2021-01-01", "2022-01-01").first().select("LC_Type1").rename("lc_modis")
        s1 = sample(ee, elev, fc, 30, ["elev_m"])
        s2 = sample(ee, lc, fc, 500, ["lc_modis"])   # mean of the (usually single) 500 m class pixel
        out = sites.set_index("site_id").join(s1).join(s2)
        out["elev_m"] = out["elev_m"].fillna(out["elev_gtopo30"])   # NASADEM stops at 60 N; GTOPO30 fills the Arctic sites
        out["lc_modis"] = out["lc_modis"].round().astype("Int64")
        out.reset_index().to_csv(f_sites, index=False)
    # embeddings
    f_emb = os.path.join(OUT, "site_year_embeddings.csv")
    if not os.path.exists(f_emb):
        parts = []
        bands = [f"A{i:02d}" for i in range(64)]
        for y in YEARS:
            log(f"embeddings at sites, {y}")
            df = sample(ee, embedding_year(ee, y).select(bands), fc, 10, bands)
            df.insert(0, "year", y)
            parts.append(df.reset_index())
        emb = pd.concat(parts).sort_values(["site_id", "year"])
        emb = emb.dropna(subset=bands)
        emb.to_csv(f_emb, index=False, float_format="%.5f")
    # monthly predictors
    f_mon = os.path.join(OUT, "site_month_predictors.csv")
    if not os.path.exists(f_mon):
        parts = []
        dm_vars = ["tmax", "tmin", "srad", "vp", "dayl", "prcp"]
        md_vars = ["ndvi_modis", "evi_modis", "lai_modis", "fpar_modis"]
        for y in YEARS:
            log(f"monthly predictors at sites, {y}")
            dm = [daymet_monthly(ee, y, m).rename([f"{v}_{m:02d}" for v in dm_vars]) for m in range(1, 13)]
            props = [f"{v}_{m:02d}" for m in range(1, 13) for v in dm_vars]
            d1 = sample(ee, ee.Image.cat(dm), fc, 1000, props)
            md = [modis_monthly(ee, y, m).rename([f"{v}_{m:02d}" for v in md_vars]) for m in range(1, 13)]
            props2 = [f"{v}_{m:02d}" for m in range(1, 13) for v in md_vars]
            d2 = sample(ee, ee.Image.cat(md), fc, 500, props2)
            ls = [landsat_ndvi_monthly(ee, y, m).rename(f"ndvi_landsat_{m:02d}") for m in range(1, 13)]
            props3 = [f"ndvi_landsat_{m:02d}" for m in range(1, 13)]
            d3 = sample(ee, ee.Image.cat(ls), fc, 30, props3)
            wide = d1.join(d2).join(d3)
            recs = []
            for sid, row in wide.iterrows():
                for m in range(1, 13):
                    rec = {"site_id": sid, "year": y, "month": m}
                    for v in dm_vars + md_vars + ["ndvi_landsat"]:
                        rec[v] = row.get(f"{v}_{m:02d}", np.nan)
                    recs.append(rec)
            parts.append(pd.DataFrame(recs))
            pd.concat(parts).to_csv(f_mon + ".partial", index=False)
        pd.concat(parts).to_csv(f_mon, index=False, float_format="%.4f")
        os.remove(f_mon + ".partial")


# --------------------------------------------------------------------------------------
# 5. Region grid export (computePixels, tiled)
# --------------------------------------------------------------------------------------
def region_grid(res):
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", REGION_CRS, always_xy=True)
    xs, ys = [], []
    for lon in (REGION_BBOX[0], REGION_BBOX[2]):
        for lat in (REGION_BBOX[1], REGION_BBOX[3]):
            x, y = tr.transform(lon, lat)
            xs.append(x)
            ys.append(y)
    x0 = np.floor(min(xs) / res) * res
    y1 = np.ceil(max(ys) / res) * res
    w = int(np.ceil((max(xs) - x0) / res))
    h = int(np.ceil((y1 - min(ys)) / res))
    return x0, y1, w, h


def fetch_grid(ee, image, bands, res, tile=200, band_chunk=64):
    """Fetch `image` (list of bands) on the region grid as a float32 array (bands, h, w).
    Requests are tiled spatially and split into chunks of bands (numpy refuses very long .npy headers)."""
    x0, y1, w, h = region_grid(res)
    out = np.full((len(bands), h, w), np.nan, dtype=np.float32)
    for b0 in range(0, len(bands), band_chunk):
        bsub = bands[b0:b0 + band_chunk]
        for r0 in range(0, h, tile):
            for c0 in range(0, w, tile):
                hh, ww = min(tile, h - r0), min(tile, w - c0)
                req = {"expression": image.select(bsub), "fileFormat": "NUMPY_NDARRAY",
                       "grid": {"dimensions": {"width": ww, "height": hh},
                                "affineTransform": {"scaleX": res, "shearX": 0, "translateX": x0 + c0 * res,
                                                    "shearY": 0, "scaleY": -res, "translateY": y1 - r0 * res},
                                "crsCode": REGION_CRS}}
                arr = retry(lambda: ee.data.computePixels(req))
                for i, b in enumerate(bsub):
                    out[b0 + i, r0:r0 + hh, c0:c0 + ww] = arr[b]
    return out, (x0, y1, w, h)


def write_tif(path, arr, res, grid, band_names, dtype="float32", scale=None, nodata=None, tags=None):
    import rasterio
    from rasterio.transform import from_origin
    x0, y1, w, h = grid
    arr = np.where(np.isfinite(arr), arr, np.nan)     # computePixels returns -inf for masked float pixels
    data = arr
    if dtype == "int16":
        data = np.where(np.isfinite(arr), np.round(arr * scale), -32768).astype(np.int16)
        nodata = -32768
    prof = dict(driver="GTiff", width=w, height=h, count=data.shape[0], dtype=dtype, crs=REGION_CRS,
                transform=from_origin(x0, y1, res, res), compress="deflate", predictor=2, nodata=nodata, tiled=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(data)
        for i, b in enumerate(band_names):
            dst.set_band_description(i + 1, b)
        t = {"scale_factor": str(1.0 / scale) if scale else "1"}
        if tags:
            t.update(tags)
        dst.update_tags(**t)
    log(f"   wrote {os.path.basename(path)}  {os.path.getsize(path) / 1e6:.1f} MB")


def export_region(ee):
    # embeddings, full 64-D for the map years
    for y in MAP_YEARS:
        p = os.path.join(REG, f"alphaearth_embedding_{y}_100m.tif")
        if os.path.exists(p):
            continue
        log(f"region: AlphaEarth {y}")
        bands = [f"A{i:02d}" for i in range(64)]
        arr, grid = fetch_grid(ee, mean_to_grid(ee, embedding_year(ee, y), 10), bands, GRID_RES)
        write_tif(p, arr, GRID_RES, grid, bands, "int16", EMB_SCALE,
                  tags={"source": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL", "footprint_mean": "10 m pixels averaged to 100 m"})
    # year-to-year similarity to the map year (dot product), all years
    p = os.path.join(REG, f"alphaearth_similarity_to_{MAP_YEARS[0]}_100m.tif")
    if not os.path.exists(p):
        log("region: similarity maps")
        # similarity of the 100 m mean embeddings (one year per request keeps Earth Engine within its memory limit)
        ref = mean_to_grid(ee, embedding_year(ee, MAP_YEARS[0]), 10)
        sims = []
        for y in YEARS:
            s_img = mean_to_grid(ee, embedding_year(ee, y), 10).multiply(ref).reduce(ee.Reducer.sum()).rename(f"sim_{y}")
            a, grid = fetch_grid(ee, s_img, [f"sim_{y}"], GRID_RES, tile=200)
            sims.append(a[0])
        write_tif(p, np.stack(sims), GRID_RES, grid, [f"sim_{y}" for y in YEARS], "float32")
    # elevation, MODIS land cover
    p = os.path.join(REG, "elevation_100m.tif")
    if not os.path.exists(p):
        log("region: elevation")
        img = mean_to_grid(ee, ee.Image("NASA/NASADEM_HGT/001").select("elevation").rename("elev_m").toFloat(), 30)
        arr, grid = fetch_grid(ee, img, ["elev_m"], GRID_RES, tile=400)
        write_tif(p, arr, GRID_RES, grid, ["elev_m"], "float32")
    p = os.path.join(REG, "modis_landcover_2021_500m.tif")
    if not os.path.exists(p):
        log("region: land cover")
        lc = ee.ImageCollection("MODIS/061/MCD12Q1").filterDate("2021-01-01", "2022-01-01").first().select("LC_Type1").rename("lc").toFloat()
        arr, grid = fetch_grid(ee, lc, ["lc"], 500, tile=400)
        write_tif(p, arr, 500, grid, ["lc_type1_igbp"], "float32")
    # Daymet monthly, all years, 1 km
    p = os.path.join(REG, "daymet_monthly_2017_2024_1km.tif")
    if not os.path.exists(p):
        log("region: Daymet monthly")
        vars_ = ["tmax", "tmin", "srad", "vp", "dayl", "prcp"]
        imgs, names = [], []
        for y in YEARS:
            for m in range(1, 13):
                imgs.append(daymet_monthly(ee, y, m).rename([f"{v}_{y}-{m:02d}" for v in vars_]))
                names += [f"{v}_{y}-{m:02d}" for v in vars_]
        arr, grid = fetch_grid(ee, ee.Image.cat(imgs), names, 1000, tile=400)
        write_tif(p, arr, 1000, grid, names, "float32")
    # MODIS vegetation monthly, all years, 500 m
    p = os.path.join(REG, "modis_vegetation_monthly_2017_2024_500m.tif")
    if not os.path.exists(p):
        log("region: MODIS NDVI/EVI/LAI/fPAR monthly")
        vars_ = ["ndvi_modis", "evi_modis", "lai_modis", "fpar_modis"]
        imgs, names = [], []
        for y in YEARS:
            for m in range(1, 13):
                imgs.append(modis_monthly(ee, y, m).rename([f"{v}_{y}-{m:02d}" for v in vars_]))
                names += [f"{v}_{y}-{m:02d}" for v in vars_]
        arr, grid = fetch_grid(ee, ee.Image.cat(imgs), names, 500, tile=400)
        write_tif(p, arr, 500, grid, names, "float32")
    # MOD16 ET and MOD17 GPP monthly mean daily rates, all years, 500 m
    for name, cid, band, scale, unit, vmax in [("mod16_et", "MODIS/061/MOD16A2GF", "ET", 0.1, "mm/day", 32700),
                                               ("mod17_gpp", "MODIS/061/MOD17A2HGF", "Gpp", 0.0001 * 1000, "gC/m2/day", 30000)]:
        p = os.path.join(REG, f"{name}_monthly_2017_2024_500m.tif")
        if os.path.exists(p):
            continue
        log(f"region: {name}")
        imgs, names = [], []
        for y in YEARS:
            for m in range(1, 13):
                imgs.append(modis_8day_to_monthly_rate(ee, cid, band, scale, y, m, vmax).rename(f"{y}-{m:02d}"))
                names.append(f"{y}-{m:02d}")
        arr, grid = fetch_grid(ee, ee.Image.cat(imgs), names, 500, tile=400)
        write_tif(p, arr, 500, grid, names, "float32", tags={"units": unit})
    # Landsat NDVI monthly and OpenET, map years, 100 m
    for y in MAP_YEARS:
        p = os.path.join(REG, f"landsat_ndvi_monthly_{y}_100m.tif")
        if not os.path.exists(p):
            log(f"region: Landsat NDVI {y}")
            geom = ee.Geometry.Rectangle(list(REGION_BBOX))
            imgs = [mean_to_grid(ee, landsat_ndvi_monthly(ee, y, m, geom).rename(f"{y}-{m:02d}"), 30) for m in range(1, 13)]
            names = [f"{y}-{m:02d}" for m in range(1, 13)]
            arr, grid = fetch_grid(ee, ee.Image.cat(imgs), names, GRID_RES, tile=400)
            write_tif(p, arr, GRID_RES, grid, names, "int16", 10000)
        p = os.path.join(REG, f"openet_ensemble_et_monthly_{y}_100m.tif")
        if not os.path.exists(p):
            log(f"region: OpenET {y}")
            names = [f"{y}-{m:02d}" for m in range(1, 13)]
            arr, grid = fetch_grid(ee, mean_to_grid(ee, openet_monthly(ee, y), 30), names, GRID_RES, tile=400)
            write_tif(p, arr, GRID_RES, grid, names, "int16", 100, tags={"units": "mm/day"})
    # small text description of the grid
    x0, y1, w, h = region_grid(GRID_RES)
    with open(os.path.join(REG, "README.md"), "w") as f:
        f.write(f"""# Region grid around US-Var (Vaira Ranch)

All rasters share the same CRS ({REGION_CRS}, UTM zone 10N) and the same extent
(lon {REGION_BBOX[0]} to {REGION_BBOX[2]}, lat {REGION_BBOX[1]} to {REGION_BBOX[3]}),
upper-left corner x={x0:.0f} m, y={y1:.0f} m; the 100 m grid is {w} x {h} pixels.
Coarser rasters (500 m MODIS, 1 km Daymet) cover the same box at their native resolution.
Band descriptions are stored in each GeoTIFF; int16 rasters carry a `scale_factor` tag.

| File | Content | Resolution |
|---|---|---|
| alphaearth_embedding_{{year}}_100m.tif | 64-band AlphaEarth annual embedding (int16 x {EMB_SCALE}) | 100 m |
| alphaearth_similarity_to_{MAP_YEARS[0]}_100m.tif | dot product of each year's 100 m mean embedding with that of {MAP_YEARS[0]} | 100 m |
| landsat_ndvi_monthly_{{year}}_100m.tif | Landsat 8/9 monthly median NDVI, 30 m pixels averaged to 100 m (int16 x 10000) | 100 m |
| openet_ensemble_et_monthly_{{year}}_100m.tif | OpenET ensemble ET, mm/day (int16 x 100) | 100 m |
| elevation_100m.tif | NASADEM elevation, m | 100 m |
| daymet_monthly_2017_2024_1km.tif | monthly mean tmax, tmin (C), srad (W/m2), vp (Pa), dayl (s), prcp (mm/day) | 1 km |
| modis_vegetation_monthly_2017_2024_500m.tif | MOD13A1 NDVI/EVI and MCD15A3H LAI/fPAR monthly means | 500 m |
| mod16_et_monthly_2017_2024_500m.tif | MOD16A2GF ET, mm/day | 500 m |
| mod17_gpp_monthly_2017_2024_500m.tif | MOD17A2HGF GPP, gC/m2/day | 500 m |
| modis_landcover_2021_500m.tif | MCD12Q1 IGBP class | 500 m |

Sources: Google / Google DeepMind AlphaEarth Foundations (CC-BY 4.0); Daymet V4 (ORNL DAAC); Landsat (USGS);
MODIS (NASA LP DAAC); OpenET ensemble v2.0; NASADEM (NASA JPL).
""")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=None, help="GEE service-account key JSON (or cloud project id for a user login)")
    ap.add_argument("--flux-dir", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--skip-region", action="store_true")
    ap.add_argument("--skip-sites", action="store_true")
    a = ap.parse_args()

    sites = build_site_table(a.flux_dir, a.metadata)
    log(f"{len(sites)} North American sites with data in {YEARS[0]}+")

    f_flux = os.path.join(OUT, "site_month_flux.csv")
    if not os.path.exists(f_flux):
        parts = []
        for i, r in enumerate(sites.itertuples()):
            d = read_flux_monthly(a.flux_dir, r)
            if d is not None and len(d):
                parts.append(d)
            if i % 25 == 0:
                log(f"flux {i}/{len(sites)}")
        flux = pd.concat(parts)
        flux.to_csv(f_flux, index=False, float_format="%.4f")
        log(f"flux rows: {len(flux)} from {flux.site_id.nunique()} sites")

    ee = ee_init(a.key)
    if not a.skip_sites:
        extract_sites(ee, sites)
    if not a.skip_region:
        export_region(ee)

    # merge flux + predictors into one tidy monthly table
    f_mon = os.path.join(OUT, "site_month_predictors.csv")
    if os.path.exists(f_mon) and os.path.exists(f_flux):
        flux = pd.read_csv(f_flux)
        pred = pd.read_csv(f_mon)
        allm = pred.merge(flux, on=["site_id", "year", "month"], how="left")
        allm.to_csv(os.path.join(OUT, "site_month_data.csv"), index=False, float_format="%.4f")
        log(f"site_month_data.csv: {len(allm)} rows, {allm.site_id.nunique()} sites, "
            f"{allm.ET_obs.notna().sum()} months with ET, {allm.GPP_obs.notna().sum()} with GPP")
    log("done")


if __name__ == "__main__":
    main()
