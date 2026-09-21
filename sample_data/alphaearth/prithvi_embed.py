"""
Step 2 of the Prithvi companion (Tutorial 13): run Prithvi-EO-2.0-300M on the HLS chips and
write compact embeddings.

Inputs : the .npz chips written by prep_prithvi_hls.py, the Prithvi weights and prithvi_mae.py
         (from https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M).
Outputs (in sample_data/alphaearth/):
    prithvi_site_month_pca64.csv     site, year, month, hls_valid_frac, P00..P63 (monthly embedding, PCA of the 1024-D token mean
                                     over the central 2 x 2 patches = 960 m around the tower)
    prithvi_site_seasonal_pca64.csv  site, year, S00..S63 (one embedding per tower-year from a 4-frame input: Jan, Apr, Jul, Oct)
    prithvi_pca.npz                  PCA means and components (1024 -> 64) for both, so the projection is reproducible
    region/prithvi_monthly_{year}_480m.tif  monthly PCA-64 embeddings on the 480 m Prithvi patch grid over the Vaira region

Usage:
    python prithvi_embed.py --chips <chip dir> --model-dir <dir with Prithvi_EO_V2_300M.pt and prithvi_mae.py> [--batch 32]
"""
import argparse, glob, json, os, sys, time, datetime
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MEAN = np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], dtype="float32")
STD = np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], dtype="float32")
CHIP, PATCH = 96, 16
GRID = CHIP // PATCH                       # 6 x 6 patches
CENTER = [(2, 2), (2, 3), (3, 2), (3, 3)]  # the four patches around the chip centre (tower)
SEASON_MONTHS = [1, 4, 7, 10]
NODATA = -9999


def log(*a):
    print(datetime.datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def load_model(model_dir, num_frames):
    import torch
    sys.path.insert(0, model_dir)
    from prithvi_mae import PrithviMAE
    cfg = json.load(open(os.path.join(model_dir, "config.json")))["pretrained_cfg"]
    cfg = {k: v for k, v in cfg.items() if k not in ("mean", "std", "bands", "origin_url", "paper_ids", "mask_ratio")}
    cfg.update(img_size=CHIP, num_frames=num_frames)
    model = PrithviMAE(**cfg)
    sd = torch.load(os.path.join(model_dir, "Prithvi_EO_V2_300M.pt"), map_location="cpu", weights_only=True)
    sd = {k: v for k, v in sd.items() if "pos_embed" not in k}          # sin-cos positions are recomputed for the 96 px grid
    missing, unexpected = model.encoder.load_state_dict({k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}, strict=False)
    log(f"model loaded (num_frames={num_frames}); missing keys: {[m for m in missing if 'pos_embed' not in m]}, unexpected: {unexpected}")
    model.eval()
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    return model.encoder


def fill_time(chips):
    """chips: (T, 6, H, W) float with NaN. Fill each NaN pixel from the nearest month with data, then the band mean."""
    T = chips.shape[0]
    f = chips.copy()
    for t in range(1, T):                       # forward fill
        m = np.isnan(f[t]); f[t][m] = f[t - 1][m]
    b = chips.copy()
    for t in range(T - 2, -1, -1):              # backward fill
        m = np.isnan(b[t]); b[t][m] = b[t + 1][m]
    out = np.where(np.isnan(f), b, np.where(np.isnan(b), f, (f + b) / 2))
    for c in range(6):                          # anything never observed -> band mean of the chip, else the global mean
        m = np.isnan(out[:, c]);
        if m.any():
            v = out[:, c][~m]
            out[:, c][m] = v.mean() if v.size else MEAN[c]
    return out


def normalise(x):
    return (x - MEAN[None, :, None, None]) / STD[None, :, None, None]


def encode(encoder, batch, num_frames):
    """batch: (B, T, 6, H, W) normalised float32 -> tokens (B, 1 + T*36, 1024) as numpy."""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 2, 1, 3, 4)))      # B, C, T, H, W
    with torch.no_grad():
        feats = encoder.forward_features(x)
    return feats[-1].numpy()


def pool_center(tokens, num_frames):
    """Mean over the four central patches (all frames): tokens (B, 1+T*36, D) -> (B, D)."""
    idx = [1 + t * GRID * GRID + r * GRID + c for t in range(num_frames) for r, c in CENTER]
    return tokens[:, idx, :].mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--skip-region", action="store_true")
    a = ap.parse_args()
    scratch = os.path.join(a.chips, "_embeddings"); os.makedirs(scratch, exist_ok=True)

    # ---------------- towers, monthly (single frame) ----------------
    f_raw = os.path.join(scratch, "site_month_center_raw.npz")
    files = sorted(f for f in glob.glob(os.path.join(a.chips, "*.npz")) if not os.path.basename(f).startswith("region_"))
    if not os.path.exists(f_raw):
        enc1 = load_model(a.model_dir, 1)
        keys, valid, embs = [], [], []
        buf, buf_keys, t0 = [], [], time.time()

        def flush():
            if not buf:
                return
            tok = encode(enc1, np.stack(buf)[:, None], 1)          # (B, 1, 6, H, W)
            embs.append(pool_center(tok, 1).astype("float16")); keys.extend(buf_keys); buf.clear(); buf_keys.clear()

        for i, f in enumerate(files):
            z = np.load(f)
            chips = z["chips"].astype("float32"); chips[chips == NODATA] = np.nan
            filled = normalise(fill_time(chips))
            for m in range(12):
                buf.append(filled[m]); buf_keys.append((str(z["site"]), int(z["year"]), m + 1)); valid.append(float(z["valid_frac"][m]))
                if len(buf) >= a.batch:
                    flush()
            if i % 50 == 0:
                log(f"monthly: {i}/{len(files)} tower-years, {time.time() - t0:.0f} s")
        flush()
        np.savez_compressed(f_raw, emb=np.concatenate(embs), keys=np.array(keys, dtype=object), valid=np.array(valid, dtype="float32"))
        log("monthly tower embeddings done")

    # ---------------- towers, seasonal (four frames in one pass) ----------------
    f_seas = os.path.join(scratch, "site_seasonal_center_raw.npz")
    if not os.path.exists(f_seas):
        enc4 = load_model(a.model_dir, 4)
        keys, embs, buf, buf_keys = [], [], [], []

        def flush4():
            if not buf:
                return
            tok = encode(enc4, np.stack(buf), 4)                    # (B, 4, 6, H, W)
            embs.append(pool_center(tok, 4).astype("float16")); keys.extend(buf_keys); buf.clear(); buf_keys.clear()

        for i, f in enumerate(files):
            z = np.load(f)
            chips = z["chips"].astype("float32"); chips[chips == NODATA] = np.nan
            filled = normalise(fill_time(chips))
            buf.append(filled[[m - 1 for m in SEASON_MONTHS]]); buf_keys.append((str(z["site"]), int(z["year"])))
            if len(buf) >= a.batch // 4:
                flush4()
        flush4()
        np.savez_compressed(f_seas, emb=np.concatenate(embs), keys=np.array(keys, dtype=object))
        log("seasonal tower embeddings done")

    # ---------------- PCA to 64 dimensions, fitted on the towers ----------------
    from sklearn.decomposition import PCA
    zm = np.load(f_raw, allow_pickle=True); zs = np.load(f_seas, allow_pickle=True)
    Xm = zm["emb"].astype("float32"); Xs = zs["emb"].astype("float32")
    pca_m = PCA(n_components=64, random_state=0).fit(Xm)
    pca_s = PCA(n_components=64, random_state=0).fit(Xs)
    log(f"PCA: monthly 64 comps explain {pca_m.explained_variance_ratio_.sum()*100:.1f}% of variance; seasonal {pca_s.explained_variance_ratio_.sum()*100:.1f}%")
    np.savez_compressed(os.path.join(HERE, "prithvi_pca.npz"), monthly_mean=pca_m.mean_, monthly_components=pca_m.components_,
                        monthly_explained=pca_m.explained_variance_ratio_, seasonal_mean=pca_s.mean_, seasonal_components=pca_s.components_,
                        seasonal_explained=pca_s.explained_variance_ratio_)
    km = pd.DataFrame(list(zm["keys"]), columns=["site_id", "year", "month"])
    dm = pd.concat([km, pd.Series(zm["valid"], name="hls_valid_frac"),
                    pd.DataFrame(pca_m.transform(Xm), columns=[f"P{i:02d}" for i in range(64)])], axis=1)
    dm.to_csv(os.path.join(HERE, "prithvi_site_month_pca64.csv"), index=False, float_format="%.4f")
    ks = pd.DataFrame(list(zs["keys"]), columns=["site_id", "year"])
    ds = pd.concat([ks, pd.DataFrame(pca_s.transform(Xs), columns=[f"S{i:02d}" for i in range(64)])], axis=1)
    ds.to_csv(os.path.join(HERE, "prithvi_site_seasonal_pca64.csv"), index=False, float_format="%.4f")
    log(f"wrote tower CSVs: {len(dm)} site-months, {len(ds)} site-years")

    # ---------------- region: monthly patch embeddings on the 480 m grid ----------------
    if a.skip_region:
        return
    import rasterio
    from rasterio.transform import from_origin
    enc1 = load_model(a.model_dir, 1)
    rfiles = sorted(glob.glob(os.path.join(a.chips, "region_*.npz")))
    years = sorted({int(np.load(f)["year"]) for f in rfiles})
    for y in years:
        out = os.path.join(HERE, "region", f"prithvi_monthly_{y}_480m.tif")
        if os.path.exists(out):
            continue
        months = [f for f in rfiles if int(np.load(f)["year"]) == y]
        z0 = np.load(months[0]); nrow, ncol = int(z0["nrow"]), int(z0["ncol"])
        # gap-fill through time per tile, as at the towers
        stack = np.stack([np.load(f)["chips"].astype("float32") for f in months])   # (12, nrow, ncol, 6, H, W)
        stack[stack == NODATA] = np.nan
        emb = np.full((12, 64, nrow * GRID, ncol * GRID), np.nan, dtype="float32")
        t0 = time.time()
        for r in range(nrow):
            for c in range(ncol):
                filled = normalise(fill_time(stack[:, r, c]))                        # (12, 6, H, W)
                tok = encode(enc1, filled[:, None], 1)                                # (12, 1+36, 1024)
                patches = tok[:, 1:, :].reshape(12, GRID, GRID, 1024)
                pcs = pca_m.transform(patches.reshape(-1, 1024)).reshape(12, GRID, GRID, 64)
                emb[:, :, r * GRID:(r + 1) * GRID, c * GRID:(c + 1) * GRID] = pcs.transpose(0, 3, 1, 2)
            log(f"region {y}: row {r + 1}/{nrow}, {time.time() - t0:.0f} s")
        data = emb.reshape(12 * 64, nrow * GRID, ncol * GRID)
        prof = dict(driver="GTiff", width=ncol * GRID, height=nrow * GRID, count=12 * 64, dtype="int16", crs=str(z0["crs"]),
                    transform=from_origin(float(z0["x0"]), float(z0["y1"]), PATCH * 30, PATCH * 30), compress="deflate", predictor=2, nodata=-32768, tiled=False)
        scale = 100.0
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(np.where(np.isfinite(data), np.round(data * scale), -32768).astype("int16"))
            for i in range(12 * 64):
                dst.set_band_description(i + 1, f"{y}-{i // 64 + 1:02d}_P{i % 64:02d}")
            dst.update_tags(scale_factor=str(1 / scale), source="Prithvi-EO-2.0-300M encoder tokens, PCA-64 fitted on tower chips",
                            patch_m="480", chip_px="96")
        log(f"wrote {os.path.basename(out)} {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
