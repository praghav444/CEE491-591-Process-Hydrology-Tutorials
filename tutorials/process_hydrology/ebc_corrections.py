# -*- coding: utf-8 -*-
"""Energy-balance-closure corrections for eddy-covariance data, for teaching.

Adapted from the research code used in Raghav et al. (2026, AmeriFlux Annual Meeting) and
Raghav & Kumar (2026, Water Resources Research). Five corrections are implemented on
half-hourly (or hourly) records; all take a DataFrame with the columns below and return the
same frame with new columns added.

Required columns (W m-2 unless noted)
    DateTime  : timestamps (start of interval)
    LE, H     : latent and sensible heat flux
    NETRAD, G : net radiation and ground heat flux
    LE_measured_flag, H_measured_flag : 1 where the flux was measured rather than gap-filled (optional)  [AEC]
    TA        : air temperature (deg C)          [FLARE]
    VPD       : vapor pressure deficit (kPa)      [PULSE, FLARE]
    PA        : air pressure (kPa)                [FLARE]
    GPP       : gross primary productivity (umol CO2 m-2 s-1)   [PULSE]
    SW_IN_POT : potential shortwave radiation (W m-2), used to define daytime [MDEBR]

Methods
    bowen_ratio_correction : Twine et al. (2000); H and LE scaled so H + LE = Rn - G, half hour by half hour
    ofc_correction         : ONEFlux EBC_CF moving-window correction (Pastorello et al., 2020), reproduced from ecbcf.c
    aec_correction         : available-energy correction (Zhang et al., 2024): a daily factor that depends on Rn - G,
                             read from the local slope of daily H + LE against Rn - G; H and LE scaled equally
    mdebr_correction       : storage-adjusted modified daytime energy-balance ratio (after Mauder et al., 2013)
    pulse_correction       : potential underlying water-use efficiency constraint (Raghav & Kumar, 2026)
    flare_correction       : surface flux equilibrium (thermodynamic Bowen ratio) constraint (Raghav & Kumar, 2026, in review)

The AEC implementation follows the authors' reference package (aec_correction, W. Zhang) step by step, in plain
numpy/pandas/scipy/statsmodels so that no extra install is needed.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr
from statsmodels.nonparametric.smoothers_lowess import lowess

# --------------------------------------------------------------------------- shared settings
EBR_REF_LOW, EBR_REF_HIGH = 0.9, 1.1      # "near-closed" reference band of the energy balance ratio
EBR_BIN_WIDTH, EBR_BIN_MIN, EBR_BIN_MAX = 0.05, 0.0, 1.5
MIN_BIN_OBS, MIN_BINS, MIN_REF_OBS = 5, 5, 5
LOESS_FRAC = 0.75
CF_MIN, CF_MAX = 1.0, 5.0                  # PULSE: LE is never decreased; factor capped at 5
_Rv, _Cp = 461.5, 1005.0                   # J kg-1 K-1


def latent_heat_J_kg(TA):
    return (2.501 - 2.361e-3 * TA) * 1e6


def potential_shortwave(datetimes, lat_deg, lon_deg, utc_offset_h):
    """Top-of-atmosphere shortwave (W m-2) at each timestamp, for defining daytime. Simple solar geometry."""
    t = pd.DatetimeIndex(datetimes)
    doy = t.dayofyear.values; hour = t.hour.values + t.minute.values / 60 + 0.25   # mid-interval for half hours
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    B = 2 * np.pi * (doy - 81) / 364
    eot = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)                 # minutes
    solar_hour = hour + (lon_deg - 15 * utc_offset_h) / 15 + eot / 60
    ha = np.deg2rad(15 * (solar_hour - 12))
    lat = np.deg2rad(lat_deg)
    cosz = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)
    return np.clip(1361 * dr * cosz, 0, None)


def ebr(df):
    ae = df["NETRAD"] - df["G"]
    return np.where(ae > 0, (df["H"] + df["LE"]) / ae, np.nan)


# --------------------------------------------------------------------------- 1. Bowen ratio (Twine et al. 2000)
def bowen_ratio_correction(df):
    """Force closure at every interval, keeping the measured Bowen ratio H/LE."""
    out = df.copy()
    ae = out["NETRAD"] - out["G"]
    beta = out["H"] / out["LE"]
    ok = (ae > 0) & (out["LE"] > 0) & (beta > -0.5)
    out["LE_BR"] = np.where(ok, ae / (1 + beta), out["LE"])
    out["H_BR"] = np.where(ok, ae - out["LE_BR"], out["H"])
    return out


# --------------------------------------------------------------------------- 2. OFC (ONEFlux ecbcf.c)
OFC_HALFWIN, OFC_ALTWIN, OFC_IQR_FACTOR, OFC_MIN_N = 15, 5, 1.5, 5


def _oneflux_percentile(sorted_vals, p):
    n = len(sorted_vals)
    idx = int(np.floor((p / 100.0) * n + 0.5 + 0.5)) - 1
    return sorted_vals[min(max(idx, 0), n - 1)]


def ofc_correction(df):
    """ONEFlux energy-balance closure factor: median of (Rn-G)/(H+LE) over a +/-15 day window of
    'stable' half hours (22:00-02:30 and 10:00-14:30), after removing outliers beyond median +/- 1.5 IQR.
    H and LE are multiplied by the same factor. Falls back to nearby factors where the window is empty."""
    out = df.copy()
    t = pd.DatetimeIndex(out["DateTime"])
    step_min = int(round(pd.Series(t).diff().median().total_seconds() / 60))
    rows_per_day = int(round(1440 / step_min)); rows_per_year = 365 * rows_per_day
    full = pd.date_range(t.min(), t.max(), freq=f"{step_min}min")
    d = out.set_index(t).reindex(full)
    N = len(full); win = OFC_HALFWIN * rows_per_day; awin = OFC_ALTWIN * rows_per_day
    ae = (d["NETRAD"] - d["G"]).values; turb = (d["H"] + d["LE"]).values
    with np.errstate(divide="ignore", invalid="ignore"):
        cf = np.where(np.isfinite(ae) & np.isfinite(turb) & (turb != 0), ae / turb, np.nan)
    val = cf[np.isfinite(cf)]
    if val.size:
        sv = np.sort(val); med = _oneflux_percentile(sv, 50); iqr = abs(_oneflux_percentile(sv, 75) - _oneflux_percentile(sv, 25))
        cf[~((cf > med - OFC_IQR_FACTOR * iqr) & (cf < med + OFC_IQR_FACTOR * iqr))] = np.nan
    hr = full.hour + full.minute / 60.0
    stable = ((hr >= 22) | (hr <= 2.5)) | ((hr >= 10) & (hr <= 14.5))
    pool = np.where(np.isfinite(cf) & stable, cf, np.nan)
    ppos = np.where(np.isfinite(pool))[0]; pval = pool[ppos]
    factor = np.full(N, np.nan); method = np.zeros(N, int)
    need = np.isfinite(d["LE"].values) | np.isfinite(d["H"].values)
    for i in np.where(need)[0]:                                   # method 1
        l = np.searchsorted(ppos, i - win, "left"); r = np.searchsorted(ppos, i + win, "right")
        if r - l >= OFC_MIN_N:
            factor[i] = _oneflux_percentile(np.sort(pval[l:r]), 50); method[i] = 1
    m1pos = np.where(method == 1)[0]; m1val = factor[m1pos]
    for i in np.where(need & (method == 0))[0]:                   # method 2
        l = np.searchsorted(m1pos, i - awin - 2, "left"); r = np.searchsorted(m1pos, i + awin, "right")
        if r - l >= 1:
            factor[i] = m1val[l:r].mean(); method[i] = 2
    for i in np.where(need & (method == 0))[0]:                   # method 3: same time +/- 1, 2 years
        acc = []
        for k in (1, 2):
            for j in (i - k * rows_per_year, i + k * rows_per_year):
                l = np.searchsorted(m1pos, j - awin, "left"); r = np.searchsorted(m1pos, j + awin, "right")
                if r - l >= 1: acc.extend(m1val[l:r])
        if acc: factor[i] = np.mean(acc); method[i] = 3
    cf_s = pd.Series(factor, index=full).reindex(t).values
    out["CF_OFC"] = cf_s
    out["LE_OFC"] = out["LE"].values * cf_s
    out["H_OFC"] = out["H"].values * cf_s
    return out


# --------------------------------------------------------------------------- 2b. AEC (Zhang et al. 2024)
AEC_MIN_GOOD = 0.5          # a day is used when at least this fraction of its intervals is measured (not gap-filled)
AEC_LOWESS_FRAC = 2 / 3     # LOWESS span for the daily H+LE vs Rn-G relation
AEC_PTS_PER_BIN = 60        # about this many days per slope bin
AEC_TAIL_N = 21             # points used to extrapolate the factor curve beyond the outermost bins
AEC_P_MAX = 0.1             # significance level for the low-energy threshold


def _aec_factor_curve(exog, endog, error_in="LeH"):
    """Correction factor as a function of daily available energy (Zhang et al., 2024).

    Daily H + LE (`endog`) is smoothed against daily Rn - G (`exog`) with LOWESS; the smoothed relation is cut into
    quantile bins of about 60 days and the slope d(H+LE)/d(Rn-G) is fitted in each bin. The inverse slope is the
    factor by which the turbulent fluxes must be scaled for the relation to have slope 1. It is interpolated to every
    day's Rn - G and extrapolated linearly in the tails. `error_in="RnG"` repeats the construction with the roles
    swapped (the error is assumed to sit in Rn - G); the two are combined by the caller."""
    m = np.isfinite(exog) & np.isfinite(endog)
    if error_in == "LeH":
        sm = lowess(endog[m], exog[m], frac=AEC_LOWESS_FRAC, return_sorted=True); x, y = sm[:, 0], sm[:, 1]
    else:
        sm = lowess(exog[m], endog[m], frac=AEC_LOWESS_FRAC, return_sorted=True); x, y = sm[:, 1], sm[:, 0]
    n_bins = max(int(m.sum()) // AEC_PTS_PER_BIN, 3)
    q = np.quantile(x, np.linspace(0, 1, n_bins))
    bx, bf = [], []
    for j in range(n_bins - 1):
        a = int(np.argmin(np.abs(x - q[j]))); b = int(np.argmin(np.abs(x - q[j + 1])))
        if b <= a + 1:
            continue
        slope = np.polyfit(x[a:b], y[a:b], 1)[0]
        if slope > 0:                                   # a non-positive slope has no meaningful inverse
            bx.append(x[a] + (x[b] - x[a]) / 2); bf.append(1.0 / slope)
    bx, bf = np.asarray(bx, float), np.asarray(bf, float)
    f = np.full_like(exog, np.nan, dtype=float)
    if len(bx) < 2:
        return f
    order = np.argsort(bx); bx, bf = bx[order], bf[order]
    inside = np.isfinite(exog) & (exog >= bx[0]) & (exog <= bx[-1])
    f[inside] = np.interp(exog[inside], bx, bf)
    srt = np.argsort(exog); xs, fs = exog[srt], f[srt]; ok = np.isfinite(xs) & np.isfinite(fs)
    lo = np.isfinite(exog) & (exog < bx[0]); hi = np.isfinite(exog) & (exog > bx[-1])
    if lo.any() and ok.sum() >= 2:
        f[lo] = np.polyval(np.polyfit(xs[ok][:AEC_TAIL_N], fs[ok][:AEC_TAIL_N], 1), exog[lo])
    if hi.any() and ok.sum() >= 2:
        f[hi] = np.polyval(np.polyfit(xs[ok][-AEC_TAIL_N:], fs[ok][-AEC_TAIL_N:], 1), exog[hi])
    return f


def aec_correction(df, min_good=AEC_MIN_GOOD):
    """Available-energy correction (Zhang et al., 2024), applied as one factor per day to H and LE.

    Steps: (1) daily means of LE, H, Rn, G from measured (not gap-filled) intervals, keeping days with at least
    `min_good` of their intervals measured; (2) a factor curve assuming the error is in H + LE and another assuming
    it is in Rn - G, combined as their geometric mean and floored at 1 (fluxes are never reduced); (3) below the daily
    available energy at which H + LE stops being significantly positively correlated with Rn - G (Pearson r > 0,
    p < 0.1 over the cumulative low-energy tail), no correction is applied; (4) each day's factor multiplies all of
    that day's half-hourly H and LE. Adds CF_AEC, LE_AEC, H_AEC; days without a factor keep the original fluxes."""
    out = df.copy()
    t = pd.DatetimeIndex(out["DateTime"]); dates = t.normalize()
    good = np.isfinite(out[["LE", "H", "NETRAD", "G"]].values).all(axis=1)
    for flag in ("LE_measured_flag", "H_measured_flag"):
        if flag in out:
            good &= out[flag].fillna(0).values.astype(bool)
    n_all = pd.Series(1, index=dates).groupby(level=0).sum()
    d = out.loc[good, ["LE", "H", "NETRAD", "G"]].copy(); d["date"] = dates[good]
    daily = d.groupby("date")[["LE", "H", "NETRAD", "G"]].mean()
    frac = d.groupby("date").size() / n_all.reindex(daily.index)
    daily = daily[frac >= min_good]
    out["CF_AEC"] = np.nan
    if len(daily) < 10:
        out["LE_AEC"], out["H_AEC"] = out["LE"], out["H"]
        return out
    exog = (daily["NETRAD"] - daily["G"].fillna(0)).values; endog = (daily["LE"] + daily["H"]).values
    f_leh = _aec_factor_curve(exog, endog, "LeH"); f_rng = _aec_factor_curve(exog, endog, "RnG")
    with np.errstate(invalid="ignore"):
        fcor = np.sqrt(f_leh * f_rng)
    fcor = np.where(np.isfinite(fcor), np.maximum(fcor, 1.0), np.nan)
    # low-energy threshold: smallest Rn - G above which the cumulative tail shows a significant positive correlation
    order = np.argsort(exog); thr = 0.0
    for i in range(3, len(order) + 1):
        r, pval = pearsonr(exog[order[:i]], endog[order[:i]])
        if np.isfinite(r) and r > 0 and pval < AEC_P_MAX:
            thr = max(float(exog[order[i - 1]]), 0.0); break
    fcor = np.where(exog > thr, fcor, 1.0)
    cf = pd.Series(fcor, index=daily.index)
    cf_hh = dates.map(cf).values.astype(float)
    out["CF_AEC"] = cf_hh; out["AEC_AE_threshold"] = thr
    use = np.isfinite(cf_hh)
    out["LE_AEC"] = np.where(use, out["LE"] * cf_hh, out["LE"])
    out["H_AEC"] = np.where(use, out["H"] * cf_hh, out["H"])
    return out


# --------------------------------------------------------------------------- 3. MDEBR
def mdebr_correction(df, potrad_thresh=50.0, ebr_min=0.3, ebr_max=1.7):
    """Daily daytime energy-balance ratio, with the nighttime residual treated as storage that is released
    during the day. Daytime H and LE are divided by the daily ratio; nighttime values are unchanged."""
    out = df.copy()
    d = out.dropna(subset=["SW_IN_POT", "NETRAD", "G", "LE", "H"]).copy()
    d["date"] = pd.DatetimeIndex(d["DateTime"]).normalize()
    is_day = d["SW_IN_POT"].values > potrad_thresh
    ae = (d["NETRAD"] - d["G"]).values; turb = (d["LE"] + d["H"]).values
    d["_ae_day"] = np.where(is_day, ae, 0.0); d["_ae_night"] = np.where(is_day, 0.0, ae)
    d["_tb_day"] = np.where(is_day, turb, 0.0); d["_tb_night"] = np.where(is_day, 0.0, turb)
    d["_n_day"] = is_day.astype(int)
    g = d.groupby("date")[["_ae_day", "_ae_night", "_tb_day", "_tb_night", "_n_day"]].sum()
    s_n = g["_ae_night"] - g["_tb_night"]                            # nighttime residual = storage
    denom = g["_ae_day"] + s_n                                        # daytime available energy minus storage release
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(np.abs(denom) > 1e-9, g["_tb_day"] / denom, np.nan)
    r = np.where(r < 0, np.nan, r); r = np.clip(r, ebr_min, ebr_max); r = np.where(g["_n_day"] > 0, r, np.nan)
    ebr_d = pd.Series(r, index=g.index)
    dates = pd.DatetimeIndex(out["DateTime"]).normalize()
    e = dates.map(ebr_d).values.astype(float)
    day = out["SW_IN_POT"].values > potrad_thresh
    use = day & np.isfinite(e) & (e > 0)
    out["EBR_d"] = np.where(day, e, np.nan)
    out["LE_MDEBR"] = np.where(use, out["LE"] / e, out["LE"])
    out["H_MDEBR"] = np.where(use, out["H"] / e, out["H"])
    return out


# --------------------------------------------------------------------------- 4. PULSE
def _quantile_reg_origin(x, y, tau=0.95):
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 10: return np.nan
    obj = lambda b: np.sum((y - b[0] * x) * (tau - ((y - b[0] * x) < 0)))
    return minimize(obj, x0=[1.0], method="Nelder-Mead").x[0]


def pulse_correction(df):
    """Potential underlying water-use efficiency (uWUEp = GPP sqrt(VPD) / ET, 95th-percentile slope) is an
    upper bound set by plant physiology and independent of the flux measurement error. Where LE is
    underestimated (EBR < 1), ET is too small and uWUEp appears inflated. The correction factor is the ratio of
    uWUEp at each closure level (LOESS of uWUEp against EBR) to uWUEp on near-closed periods (EBR 0.9-1.1)."""
    out = df.copy()
    for c in ("CF_PULSE", "uWUEp_ref", "uWUEp_pred"): out[c] = np.nan
    out["LE_PULSE"] = out["LE"].values.copy()
    dt = pd.DatetimeIndex(out["DateTime"])
    TA, LE, H, Rn, G = (out[c].values for c in ("TA", "LE", "H", "NETRAD", "G"))
    GPP_gC = out["GPP"].values * 12.011e-6 * 86400.0                 # gC m-2 day-1 equivalent
    lv = latent_heat_J_kg(TA)
    ET = np.where((lv > 0) & (LE >= 0), LE * 86400.0 / lv, np.nan)  # mm day-1 equivalent
    signal = GPP_gC * np.sqrt(np.clip(out["VPD"].values, 0, None))
    AE = Rn - G
    with np.errstate(divide="ignore", invalid="ignore"):
        EBR = np.where(AE > 0, (H + LE) / AE, np.nan)
    gpp_day = pd.Series(GPP_gC).groupby(dt.normalize()).transform("mean").values
    season = gpp_day > 0.10 * np.nanquantile(gpp_day, 0.95)
    bins = np.arange(EBR_BIN_MIN, EBR_BIN_MAX + EBR_BIN_WIDTH, EBR_BIN_WIDTH); centers = 0.5 * (bins[:-1] + bins[1:])
    for yr in sorted(dt.year.unique()):
        idx = dt.year.values == yr
        e, et, sg, sea, le, lvy = EBR[idx], ET[idx], signal[idx], season[idx], LE[idx], lv[idx]
        ref = (e >= EBR_REF_LOW) & (e <= EBR_REF_HIGH) & sea & (et > 0) & np.isfinite(sg) & (sg > 0)
        if ref.sum() < MIN_REF_OBS: continue
        u_ref = _quantile_reg_origin(et[ref], sg[ref])
        if not np.isfinite(u_ref) or u_ref <= 0: continue
        valid = np.isfinite(e) & (et > 0) & np.isfinite(sg) & (sg > 0)
        gc, gu = [], []
        for i, bc in enumerate(centers):
            inb = valid & (e >= bins[i]) & (e < bins[i + 1])
            if inb.sum() >= MIN_BIN_OBS:
                u = _quantile_reg_origin(et[inb], sg[inb])
                if np.isfinite(u): gc.append(bc); gu.append(u)
        if len(gc) < MIN_BINS: continue
        fit = lowess(np.array(gu), np.array(gc), frac=LOESS_FRAC, return_sorted=True, it=3)
        u_pred = np.interp(e, fit[:, 0], fit[:, 1], left=np.nan, right=np.nan)
        u_pred = np.where((e >= fit[:, 0].min()) & (e <= fit[:, 0].max()), u_pred, np.nan)
        cf = u_pred / u_ref
        cf_c = np.clip(cf, CF_MIN, CF_MAX)
        et_c = np.where(np.isfinite(cf), np.where(cf >= CF_MIN, et * cf_c, et), np.nan)
        closed = np.isfinite(e) & (e >= 1.0); et_c[closed] = et[closed]
        le_c = np.where(np.isfinite(et_c), et_c * lvy / 86400.0, le)   # undefined -> no correction
        out.loc[idx, "LE_PULSE"] = le_c
        out.loc[idx, "CF_PULSE"] = np.where(closed, 1.0, np.clip(cf, CF_MIN, None))
        out.loc[idx, "uWUEp_ref"] = u_ref; out.loc[idx, "uWUEp_pred"] = u_pred
    out["EBR"] = EBR
    return out


# --------------------------------------------------------------------------- 5. FLARE (surface flux equilibrium)
def _loess_signal(ebr_arr, signal_arr):
    bins = np.arange(EBR_BIN_MIN, EBR_BIN_MAX + EBR_BIN_WIDTH, EBR_BIN_WIDTH); centers = 0.5 * (bins[:-1] + bins[1:])
    n = len(ebr_arr); sp = np.full(n, np.nan)
    ref = (ebr_arr >= EBR_REF_LOW) & (ebr_arr <= EBR_REF_HIGH) & np.isfinite(signal_arr) & (signal_arr > 0)
    if ref.sum() < MIN_REF_OBS: return np.nan, sp
    valid = np.isfinite(ebr_arr) & np.isfinite(signal_arr) & (signal_arr > 0)
    gc, ga = [], []
    for i, bc in enumerate(centers):
        inb = valid & (ebr_arr >= bins[i]) & (ebr_arr < bins[i + 1])
        if inb.sum() >= MIN_BIN_OBS: gc.append(bc); ga.append(np.median(signal_arr[inb]))
    if len(gc) < MIN_BINS: return np.nan, sp
    fit = lowess(np.array(ga), np.array(gc), frac=LOESS_FRAC, return_sorted=True, it=3)
    sig_ref = float(np.interp(1.0, fit[:, 0], fit[:, 1]))
    if not np.isfinite(sig_ref) or sig_ref <= 0: return np.nan, sp
    sp[:] = np.where(np.isfinite(ebr_arr), np.interp(ebr_arr, fit[:, 0], fit[:, 1]), np.nan)
    return sig_ref, sp


def flare_correction(df):
    """Thermodynamic equilibrium Bowen ratio B = Rv cp T^2 / (lv^2 q) gives an equilibrium partition of
    available energy (H_eq, LE_eq) that does not depend on the flux measurement. The signal H_eq / (AE - LE)
    falls below its near-closure value when LE is underestimated; the fraction of the closure gap given to LE is
    1 - signal/reference, bounded by the gap itself and by LE_eq."""
    out = df.copy()
    TA, VPD, PA, LE, H, Rn, G = (out[c].values for c in ("TA", "VPD", "PA", "LE", "H", "NETRAD", "G"))
    T_K = TA + 273.15; lv = latent_heat_J_kg(TA)
    es = 0.6112 * np.exp(17.67 * TA / (TA + 243.5)); ea = np.clip(es - VPD, 1e-6, None)
    q = (0.622 * ea) / (PA - 0.378 * ea); q = np.where(q > 1e-6, q, np.nan)
    B = _Rv * _Cp * T_K ** 2 / (lv ** 2 * q)
    AE = Rn - G
    LE_eq = np.where(B > 0, AE / (1.0 + B), np.nan); H_eq = np.where(B > 0, AE * B / (1.0 + B), np.nan)
    H_res = AE - LE
    sig = np.where((AE > 0) & (H_res > 0) & np.isfinite(H_eq), H_eq / H_res, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        EBR = np.where(AE > 0, (H + LE) / AE, np.nan)
    out["B_eq"], out["LE_eq"], out["EBR"], out["sig_FLARE"] = B, LE_eq, EBR, sig
    if "GPP" in out and out["GPP"].notna().any():
        gpp = out["GPP"].values * 12.011e-6 * 86400; pos = gpp[gpp > 0]
        season = gpp > (np.nanpercentile(pos, 10) if len(pos) else 0.0)
    else:
        season = np.ones(len(out), bool)
    out["LE_FLARE"] = LE.copy(); out["CF_FLARE"] = np.nan; out["r_FLARE"] = np.nan
    dt = pd.DatetimeIndex(out["DateTime"])
    for yr in sorted(dt.year.unique()):
        idx = dt.year.values == yr
        e, s, le, h, ae, leq = EBR[idx], sig[idx], LE[idx], H[idx], AE[idx], LE_eq[idx]
        sig_ref, sig_pred = _loess_signal(e, np.where(season[idx], s, np.nan))
        if not np.isfinite(sig_ref): continue
        r = np.where(np.isfinite(sig_pred) & (sig_pred > 0), sig_pred / sig_ref, np.nan)
        apply = np.isfinite(r) & (r < 1.0) & (ae > 0) & (le < ae)
        gap = np.clip(ae - h - le, 0.0, None)
        cap = np.where(np.isfinite(leq), np.maximum(le, leq), np.inf)
        le_c = np.where(apply, np.minimum(le + (1.0 - r) * gap, cap), le)
        le_c = np.maximum(le_c, le)
        out.loc[idx, "LE_FLARE"] = le_c
        with np.errstate(divide="ignore", invalid="ignore"):
            out.loc[idx, "CF_FLARE"] = np.where(np.isfinite(le) & (le > 5.0), le_c / le, np.nan)
        out.loc[idx, "r_FLARE"] = r
    return out


# --------------------------------------------------------------------------- convenience
def apply_all(df):
    out = bowen_ratio_correction(df)
    out = ofc_correction(out)
    out = aec_correction(out)
    out = mdebr_correction(out)
    out = pulse_correction(out)
    out = flare_correction(out)
    return out


def daily_et_mm(df, cols=("LE", "LE_BR", "LE_OFC", "LE_AEC", "LE_MDEBR", "LE_PULSE", "LE_FLARE"), min_completeness=0.75):
    """Daily ET (mm/day) from half-hourly W m-2, requiring at least `min_completeness` of the day."""
    d = df.copy(); t = pd.DatetimeIndex(d["DateTime"]); d["date"] = t.normalize()
    step_min = int(round(pd.Series(t).diff().median().total_seconds() / 60)); expected = int(round(1440 / step_min))
    ta = d.groupby("date")["TA"].mean(); lv = 2.501 - 0.002361 * ta
    res = pd.DataFrame(index=ta.index)
    for c in cols:
        if c not in d: continue
        g = d.groupby("date")[c].agg(["sum", "count"])
        et = (g["sum"] / expected) * 86400 / (lv.reindex(g.index) * 1e6)
        res[c.replace("LE", "ET")] = et.where(g["count"] / expected >= min_completeness)
    return res
