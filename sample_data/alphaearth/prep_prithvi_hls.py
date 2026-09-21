"""
Step 1 of the Prithvi companion (Tutorial 13): download monthly HLS chips from Earth Engine.

For every tower-year that has flux data in site_month_data.csv, and for the Vaira region in the
map years, this script builds monthly cloud-masked median composites of the six HLS bands that
Prithvi-EO-2.0 expects (Blue, Green, Red, NIR-narrow, SWIR1, SWIR2) on 96 x 96 pixel chips at
30 m (2.88 km) and stores them as int16 HLS reflectance x 10000 in .npz files.

Usage:
    python prep_prithvi_hls.py --key gee-key.json --out <chip directory> [--workers 8] [--skip-region]
"""
import argparse, json, os, time, datetime, calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CHIP = 96                     # pixels; 6 x 6 Prithvi patches of 16 px (480 m)
RES = 30
NODATA = -9999
BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
L30 = {"B2": "blue", "B3": "green", "B4": "red", "B5": "nir", "B6": "swir1", "B7": "swir2"}
S30 = {"B2": "blue", "B3": "green", "B4": "red", "B8A": "nir", "B11": "swir1", "B12": "swir2"}
YEARS = list(range(2017, 2025))
MAP_YEARS = [2021, 2023]
REGION_BBOX = (-121.25, 38.20, -120.80, 38.56)
REGION_CRS = "EPSG:32610"


def log(*a):
    print(datetime.datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def ee_init(key):
    import ee
    sa = json.load(open(key))["client_email"]
    ee.Initialize(ee.ServiceAccountCredentials(sa, key), opt_url="https://earthengine-highvolume.googleapis.com")
    return ee


def hls_monthly(ee, y, m, region):
    """Median of cloud-masked HLS L30 + S30 scenes for one month, six bands, reflectance x 10000."""
    start = ee.Date.fromYMD(y, m, 1)
    end = start.advance(1, "month")

    def prep(img, mapping):
        fm = img.select("Fmask")
        clear = fm.bitwiseAnd(2).eq(0).And(fm.bitwiseAnd(4).eq(0)).And(fm.bitwiseAnd(8).eq(0))   # cloud, adjacent, shadow
        return img.select(list(mapping.keys()), list(mapping.values())).updateMask(clear)

    l30 = ee.ImageCollection("NASA/HLS/HLSL30/v002").filterDate(start, end).filterBounds(region).map(lambda i: prep(i, L30))
    s30 = ee.ImageCollection("NASA/HLS/HLSS30/v002").filterDate(start, end).filterBounds(region).map(lambda i: prep(i, S30))
    coll = l30.merge(s30)
    img = ee.Image(ee.Algorithms.If(coll.size().gt(0), coll.median(), ee.Image.constant([0] * 6).rename(BANDS).updateMask(0)))
    return img.select(BANDS).multiply(10000).toFloat()


def utm_crs(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def chip_grid_for_site(lon, lat):
    from pyproj import Transformer
    crs = utm_crs(lon, lat)
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    x0 = np.round(x / RES) * RES - CHIP // 2 * RES
    y1 = np.round(y / RES) * RES + CHIP // 2 * RES
    return crs, x0, y1


def fetch_chip(ee, image, crs, x0, y1, w=CHIP, h=CHIP):
    req = {"expression": image, "fileFormat": "NUMPY_NDARRAY",
           "grid": {"dimensions": {"width": w, "height": h},
                    "affineTransform": {"scaleX": RES, "shearX": 0, "translateX": float(x0), "shearY": 0, "scaleY": -RES, "translateY": float(y1)},
                    "crsCode": crs}}
    for i in range(4):
        try:
            arr = ee.data.computePixels(req)
            out = np.stack([arr[b] for b in BANDS]).astype("float32")
            out[~np.isfinite(out)] = np.nan
            return out
        except Exception as e:  # noqa
            if i == 3:
                raise
            time.sleep(10 * (i + 1))


def to_int16(a):
    b = np.where(np.isfinite(a), np.round(a), NODATA)
    return np.clip(b, -32768, 32767).astype(np.int16)


def do_site_year(ee, site, lon, lat, year, out_dir):
    f = os.path.join(out_dir, f"{site}_{year}.npz")
    if os.path.exists(f):
        return "skip"
    crs, x0, y1 = chip_grid_for_site(lon, lat)
    region = ee.Geometry.Rectangle([lon - 0.03, lat - 0.03, lon + 0.03, lat + 0.03])
    chips = np.full((12, 6, CHIP, CHIP), np.nan, dtype="float32")
    for m in range(1, 13):
        chips[m - 1] = fetch_chip(ee, hls_monthly(ee, year, m, region), crs, x0, y1)
    valid = np.isfinite(chips[:, 0]).mean(axis=(1, 2))
    np.savez_compressed(f, chips=to_int16(chips), valid_frac=valid.astype("float32"), crs=crs, x0=x0, y1=y1, year=year, site=site)
    return "ok"


def region_tiles():
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", REGION_CRS, always_xy=True)
    xs, ys = zip(*[tr.transform(lon, lat) for lon in (REGION_BBOX[0], REGION_BBOX[2]) for lat in (REGION_BBOX[1], REGION_BBOX[3])])
    x0 = np.floor(min(xs) / (CHIP * RES)) * (CHIP * RES)
    y1 = np.ceil(max(ys) / (CHIP * RES)) * (CHIP * RES)
    ncol = int(np.ceil((max(xs) - x0) / (CHIP * RES)))
    nrow = int(np.ceil((y1 - min(ys)) / (CHIP * RES)))
    return x0, y1, nrow, ncol


def do_region_month(ee, year, month, out_dir):
    f = os.path.join(out_dir, f"region_{year}_{month:02d}.npz")
    if os.path.exists(f):
        return "skip"
    x0, y1, nrow, ncol = region_tiles()
    geom = ee.Geometry.Rectangle(list(REGION_BBOX)).buffer(5000)
    img = hls_monthly(ee, year, month, geom)
    chips = np.full((nrow, ncol, 6, CHIP, CHIP), np.nan, dtype="float32")
    for r in range(nrow):
        for c in range(ncol):
            chips[r, c] = fetch_chip(ee, img, REGION_CRS, x0 + c * CHIP * RES, y1 - r * CHIP * RES)
    valid = np.isfinite(chips[:, :, 0]).mean(axis=(2, 3))
    np.savez_compressed(f, chips=to_int16(chips), valid_frac=valid.astype("float32"), crs=REGION_CRS, x0=x0, y1=y1, nrow=nrow, ncol=ncol, year=year, month=month)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-region", action="store_true")
    ap.add_argument("--skip-sites", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ee = ee_init(a.key)

    sites = pd.read_csv(os.path.join(HERE, "sites.csv")).set_index("site_id")
    mon = pd.read_csv(os.path.join(HERE, "site_month_data.csv"))
    has = mon[(mon.LE_F_MDS_QC >= 0.7) & mon.ET_obs_corr.notna() | (mon.NEE_VUT_REF_QC >= 0.7) & mon.GPP_obs.notna()]
    site_years = has[["site_id", "year"]].drop_duplicates().sort_values(["site_id", "year"])
    log(f"{len(site_years)} tower-years with flux data -> {len(site_years) * 12} monthly chips")

    if not a.skip_sites:
        jobs = [(r.site_id, sites.loc[r.site_id, "lon"], sites.loc[r.site_id, "lat"], int(r.year)) for r in site_years.itertuples()]
        done = 0
        with ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(do_site_year, ee, *j, a.out): j for j in jobs}
            for fut in as_completed(futs):
                j = futs[fut]
                try:
                    fut.result()
                except Exception as e:  # noqa
                    log(f"FAILED {j[0]} {j[3]}: {str(e)[:120]}")
                done += 1
                if done % 50 == 0:
                    log(f"sites: {done}/{len(jobs)}")
        log("tower chips done")

    if not a.skip_region:
        x0, y1, nrow, ncol = region_tiles()
        log(f"region: {nrow} x {ncol} chips per month")
        jobs = [(y, m) for y in MAP_YEARS for m in range(1, 13)]
        with ThreadPoolExecutor(min(a.workers, 4)) as ex:
            futs = {ex.submit(do_region_month, ee, y, m, a.out): (y, m) for y, m in jobs}
            for fut in as_completed(futs):
                try:
                    fut.result(); log(f"region {futs[fut]} done")
                except Exception as e:  # noqa
                    log(f"FAILED region {futs[fut]}: {str(e)[:120]}")
        log("region chips done")


if __name__ == "__main__":
    main()
