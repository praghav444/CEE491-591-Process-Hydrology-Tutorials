from numpy import cos, exp, isnan, log, pi, sin, tan
from pandas import Series, to_numeric
from xarray import DataArray
from numpy import arccos, clip, nanmax, where
import numpy as np
import pandas as pd
from meteo_utils import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
def fill_nan(x, val):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.fillna(val)
    elif x is None:
        return val
    else:
        try:
            return val if np.isnan(x) else x
        except TypeError:
            return x
# Priestley-Taylor
def calc_PET_priestley_taylor(tmean,rn,g,pressure,alpha):
    """Potential evapotranspiration calculated according to
    :cite:t:`priestley_assessment_1972`.

    Parameters
    ----------
    tmean: pandas.Series or xarray.DataArray
        air temperature [degC].
    rn: float or pandas.Series or xarray.DataArray
        net radiation [W m-2].
    g: float or pandas.Series or xarray.DataArray
        soil heat flux [W m-2].
    pressure: float or pandas.Series or xarray.DataArray
        atmospheric pressure [kPa].
    alpha: float
        calibration coefficient [-].

    Returns
    -------
    pandas.Series or xarray.DataArray containing the calculated pootential
    evapotranspiration [W m-2].
    
        Notes
    -----

    .. math:: PET = \\frac{\\alpha_{PT} \\Delta (R_n-G)}
        {\\(\\Delta +\\gamma)}
    """
    gamma = calc_psy(pressure)
    dlt = calc_vpc(tmean)
    pet = (alpha * dlt * (rn - g)) / ((dlt + gamma))
    pet = clip_zeros(pet, clip_zero=True)
    return pet
#----------------------------------------------------------------------
def obj_function_PT(alpha, tmean, rn, g, pressure, obs_PET):
    # Calculate simulated PET using the Priestley-Taylor method with the current alpha
    sim_PET = calc_PET_priestley_taylor(tmean, rn, g, pressure, alpha)
    error = np.sqrt(np.mean((obs_PET - sim_PET) ** 2))
    return error
def cal_alpha_from_PET(tmean, rn, g, pressure, obs_PET):
    # alpha_PT = PET*(delta + gamma)/(delta * (Rn-G))
    gamma = calc_psy(pressure)
    dlt = calc_vpc(tmean)
    alpha_PT = obs_PET*(dlt + gamma)/(dlt*(rn-g))
    return alpha_PT
def tilted_loss(y_true, y_pred, tau):
    """Quantile (tilted absolute) loss function."""
    error = y_true - y_pred
    return np.sum((tau - (error < 0)) * error)
def obj_function_PT_quant(alpha, tmean, rn, g, pressure, obs_PET, tau=0.95):
    # Simulate PET with current alpha
    sim_PET = calc_PET_priestley_taylor(tmean, rn, g, pressure, alpha)
    
    # Apply quantile (tilted) loss instead of RMSE or MAE
    loss = tilted_loss(obs_PET, sim_PET, tau)
    return loss
#----------------------------------------------------------------------
def inv_PM_for_gs(LE, tmean, wind, USTAR, rn, g, vpd, pressure, zm, zh, ra_method=1):
    # Invert Penman-Monteith to estimate surface conductance (Gs, mmol m⁻² s⁻¹)
    
    gamma = calc_psy(pressure)            # Psychrometric constant
    dlt = calc_vpc(tmean)                 # Slope of saturation vapor pressure curve
    Cp = 1013                             # Specific heat of air at constant pressure (J/kg/°C)
    es = calc_e0(tmean)                   # Saturation vapor pressure (kPa)
    ea = es - vpd                         # Actual vapor pressure (kPa)
    rho_a = calc_rho(pressure, tmean, ea) # Air density (kg/m³)

    res_a = calc_res_aero(wind=wind, USTAR=USTAR,ra_method=ra_method, croph=zh, zw=zm, zh=zm)  # Aerodynamic resistance (s/m)
    res_a[(res_a >= 6000) | (res_a < 0)] = np.nan
    # Calculate surface resistance (Rs, s/m)
    numerator = dlt * (rn - g) + rho_a * Cp / res_a * vpd
    denominator = LE
    Rs = ((numerator / denominator) - (dlt + gamma)) / (gamma / res_a)
    Rs[(Rs >= 6000) | (Rs < 0)] = np.nan

    # Convert to surface conductance (mmol m⁻² s⁻¹)
    Tk = tmean + 273.15  # Convert to Kelvin
    Gs_mmol = (1 / Rs) * (gamma * 1e3 / 0.000665) / (8.3143 * Tk) * 1000

    return Gs_mmol, res_a

import numpy as np

def compute_FAO_PET(Ta2M, Pa, uz, Zum,
                    QV2M=None, VPD=None,
                    H=None, LE=None, SWnet=None, LWnet=None, Rn = None, G=None, Ca=None):
    """
    Compute Potential Evapotranspiration (PET) using the FAO Penman–Monteith equation.

    Parameters
    ----------
    Ta2M : array-like
        Air temperature at 2 m (°C)
    Pa : array-like
        Surface pressure (kPa or Pa)
    uz : array-like
        Wind speed at measurement height (m/s)
    Zum : float
        Height of wind measurement (m)
    QV2M : array-like, optional
        Specific humidity (kg/kg)
    VPD : array-like, optional
        Vapor pressure deficit (kPa)
    H, LE, SWnet, LWnet, G : array-like, optional
        Energy flux components (W/m²)
    Ca : array-like, optional
        Atmospheric CO2 concentration (ppm)
    Returns
    -------
    PET : np.ndarray
        Potential evapotranspiration (W/m²)
    """

    T = np.asarray(Ta2M, dtype=float)
    P = np.asarray(Pa, dtype=float)

    # --- Ensure Temperature is in °C
    if np.nanmean(T) > 100:  # likely in K
        T = T - 273.15
    # --- Ensure pressure is in kPa
    if np.nanmean(P) > 200:  # likely in Pa
        P = P / 1000.0

    # --- Available energy (W/m²)
    if SWnet is not None and LWnet is not None and G is not None:
        A = SWnet + LWnet - G
    if Rn is not None and G is not None:
        A = Rn - G
    elif H is not None and LE is not None:
        A = H + LE
    else:
        raise ValueError("Provide either (SWnet, LWnet, G) or (H, LE) for available energy.")

    # --- Wind speed correction to 2 m (FAO-56 standard)
    u2 = uz #* (4.87 / np.log(67.8 * Zum - 5.42))

    # --- Convert A from W/m² → MJ/m²/day
    A_MJ = A * 86400 / 1e6

    # --- Psychrometric constant (kPa/°C)
    gamma = 0.000665 * P

    # --- Slope of saturation vapor pressure curve (kPa/°C)
    delta = 4098 * (0.6108 * np.exp(17.27 * T / (T + 237.3))) / (T + 237.3) ** 2

    # --- Saturation vapor pressure (kPa)
    es = 0.6108 * np.exp(17.27 * T / (T + 237.3))

    # --- Actual vapor pressure (kPa)
    if VPD is None:
        if QV2M is None:
            raise ValueError("Either VPD or QV2M must be provided.")
        e = QV2M * P / (0.622 + 0.378 * QV2M)
        VPD = es - e

    # --- FAO Penman–Monteith (mm/day)
    PET_mm_day = (
        0.408 * delta * A_MJ
        + gamma * 900 * u2 * VPD / (T + 273)
    ) / (delta + gamma * (1 + u2*(0.34 + 0.00024*(Ca - 300)))) # Includes CO2 effect

    # --- Convert mm/day → W/m² using temperature-dependent λ
    λ = 2.501e6 - 2.361e3 * T  # J/kg
    PET = PET_mm_day * λ / 86400.0

    return PET

#----------------------------------------------------------------------
# Penman-Monteith
def calc_pet_penman_monteith_raghav(tmean,wind,rn,g,Rg,vpd,pressure,zm,zh,LAI,USTAR,CO2=300,ra_method=0,lai_eff=0,r_l_max=5000,r_l_min=100,srs=0.0009,hs=50,Tref=25,Rgl=100,clip_zero=True):
    """Potential evapotranspiration calculated according to
    :cite:t:`monteith_evaporation_1965`.

    Parameters
    ----------
    tmean: float or xarray.DataArray
        air temperature [degC].
    wind: float or pandas.Series or xarray.DataArray
        wind speed [m/s].
    rn: float or pandas.Series or xarray.DataArray
        net radiation [W m-2].
    g: float or pandas.Series or xarray.DataArray
        soil heat flux [W m-2].
    Rg: float or pandas.Series or xarray.DataArray
        Solar Radiation [W m-2].
    vpd: float or pandas.Series or xarray.DataArray
        vapor pressure deficit [kPa].
    pressure: float or xarray.DataArray
        atmospheric pressure [kPa].
    zm: float or xarray.DataArray
        height of wind or humidity measurement [m].
    zh: float or xarray.DataArray
        vegetation height [m].
    LAI: float or pandas.Series or xarray.DataArray
        leaf area index [-].
    CO2: float or pandas.Series or xarray.DataArray
        CO2 concentration [ppm].
    r_l_min: pandas.Series or float
        minimum bulk stomatal resistance (a calibration parameter) [s m-1].
    r_l_max: pandas.Series or float
        maximum bulk stomatal resistance [=5000 s m-1].
    ra_method: float, optional
        0 => ra = 208/wind
        1 => ra is calculated based on equation 36 in FAO (1990), ANNEX V.
        2 => ra = u/u*^2 + 6.2*u*^(0.67)
    lai_eff: float
        0 => LAI_eff = 0.5 * LAI
        1 => LAI_eff = lai / (0.3 * lai + 1.2)
        2 => LAI_eff = 0.5 * LAI; (LAI>4=4)
        3 => see :cite:t:`zhang_comparison_2008`.
    srs: float or pandas.Series or xarray.DataArray
        Relative sensitivity of rl to ?[CO2].
    hs: float or pandas.Series or xarray.DataArray
        Parameter in the vapor pressure resistance factor,
    Tref: float or pandas.Series or xarray.DataArray
        Optimum air temperature for transpiration [degC].
    Rgl: float or pandas.Series or xarray.DataArray
        solar radiation threshold for which resistance factor Rsr is about to double its minimum value [W m-2].
    clip_zero: bool, optional
        if True, replace all negative values with 0.
    Returns
    -------
    pandas.Series or xarray.DataArray containing the calculated potential
    evapotranspiration [W m-2].

    Notes
    -----

    Following :cite:t:`monteith_evaporation_1965`, :cite:t:`allen_crop_1998`,
    :cite:t:`zhang_comparison_2008`, :cite:t:`schymanski_leaf-scale_2017` and
    :cite:t:`yang_hydrologic_2019`.

    .. math:: PET = \\frac{\\Delta (R_{n}-G)+ \\rho_a c_p
        \\frac{e_s-e_a}{r_a}}{\\(\\Delta +\\gamma(1+\\frac{r_s}{r_a}))}

    , where

    .. math:: r_s = r_l_min / ( LAI_{eff} * f_{CO2} * f_{q} * f_{Ta} * f_(Rg) )

    .. math:: f_{CO2} = 1/(1+srs*(CO_2-300))
    .. math:: f_{q} = 1/(1+hs*(Qs-Qa))
    .. math:: f_{Ta} = 1-0.0016(Tref-Ta)**2
    .. math:: f_{Rg} = (r_l_min/r_l_max + f)/(1+f) with f = 0.55*(Rg/Rgl)*(2/LAI)

    ra_method == 0:

    .. math:: r_a = \\frac{208}{u_2}

    ra_method == 1:

    .. math:: r_a = log(\\frac{(zw - d)}{zom}) *
        \\frac{log(\\frac{(zh - d)}{zoh})}{(0.41^2)u_2}

    """
    gamma = calc_psy(pressure)
    dlt = calc_vpc(tmean)
    es = calc_e0(tmean) # kPa
    ea = es - vpd   # kPa
    Qs = 0.622*es/(pressure - 0.378*es)  # Sp. Humidity (g kg-1)
    Qa = 0.622*ea/(pressure - 0.378*ea)
    # Aerodynamic Resistance (Ra; s m-1)
    res_a = calc_res_aero(wind=wind,USTAR=USTAR, ra_method=ra_method, croph=zh, zw=zm, zh=zm)
    res_a = fill_nan(res_a, 100)
    # Surface Resistance (Rs; s m-1)
    f_co2 = 1/(1 + srs * (CO2 - 300))                    # <<<<--- CO2 stress factor
    f_q = 1/(1+hs*(Qs-Qa))                               # <<<<--- Sp. Humidity Stress factor
    f_Ta = 1-0.00016*(Tref-tmean)**2                       # <<<<--- Temperature Stress Factor
    f_Rg_1 = 0.55*(Rg/Rgl)*(2/LAI)
    f_Rg = (r_l_min/r_l_max + f_Rg_1)/(1+f_Rg_1)         # <<<<--- Radiation Stress Factor

    res_s = r_l_min/(LAI * fill_nan(f_co2, 1) * fill_nan(f_q, 1) * fill_nan(f_Ta, 1) * fill_nan(f_Rg, 1))
    #res_s = r_l_min/LAI
    res_s = res_s.clip(lower=10, upper=5000)
    
    gamma1 = gamma * (1 + res_s / res_a)
    
    rho_a = calc_rho(pressure, tmean, ea)
    
    den = (dlt + gamma1)  # Denominator
    
    num1 = dlt * (rn - g) / den
    CP = 1013  # J kg-1 degC-1 
    num2 = rho_a * CP * vpd / res_a / den
    pet = num1 + num2
    pet = clip_zeros(pet, clip_zero)
    return pet
def obj_function_PM_raghav(r_l_min,tmean,wind,rn,g,Rg,vpd,pressure,zm,zh,LAI,USTAR,CO2,hs,Rgl,obs_PET):
    # Calculate simulated PET using the Penman-Monteith method with the current r_l_min
    sim_PET = calc_pet_penman_monteith_raghav(tmean=tmean, wind=wind, rn=rn, g=g, Rg=Rg, vpd=vpd, pressure=pressure, zm=zm, zh=zh, LAI=LAI, USTAR=USTAR, CO2=CO2, r_l_min=r_l_min, hs=hs, Rgl=Rgl)
    error = calc_error(obs_PET, sim_PET)
    return error
# Penman-Monteith
def calc_pet_penman_monteith(tmean,wind,rn,g,vpd,pressure,zm,lai,croph=0.12,r_l_min=100,r_s=None,ra_method=1,a_sh=1,a_s=1,lai_eff=0,srs=0.0009,co2=300,clip_zero=True):
    """Potential evapotranspiration calculated according to
    :cite:t:`monteith_evaporation_1965`.

    Parameters
    ----------
    tmean: float or xarray.DataArray
        air temperature [degC].
    wind: float or pandas.Series or xarray.DataArray
        wind speed [m/s].
    rn: float or pandas.Series or xarray.DataArray
        net radiation [W m-2].
    g: float or pandas.Series or xarray.DataArray
        soil heat flux [W m-2].
    vpd: float or pandas.Series or xarray.DataArray
        vapor pressure deficit [kPa].
    pressure: float or xarray.DataArray
        atmospheric pressure [kPa].
    zm: float or xarray.DataArray
        height of wind or humidity measurement [m].
    lai: float or pandas.Series or xarray.DataArray
        leaf area index [-].
    croph: float or pandas.Series or xarray.DataArray
        crop or vegetation height [m].
    r_l_min: pandas.Series or float
        minimum bulk stomatal resistance (a calibration parameter) [s m-1].
    ra_method: float, optional
        0 => ra = 208/wind
        1 => ra is calculated based on equation 36 in FAO (1990), ANNEX V.
    a_s: float, optional
        Fraction of one-sided leaf area covered by stomata (1 if stomata are 1
        on one side only, 2 if they are on both sides).
    a_sh: float, optional
        Fraction of projected area exchanging sensible heat with the air (2).
    lai_eff: float, optional
        0 => LAI_eff = 0.5 * LAI
        1 => LAI_eff = lai / (0.3 * lai + 1.2)
        2 => LAI_eff = 0.5 * LAI; (LAI>4=4)
        3 => see :cite:t:`zhang_comparison_2008`.
    srs: float or pandas.Series or xarray.DataArray, optional
        Relative sensitivity of rl to ?[CO2].
    co2: float or pandas.Series or xarray.DataArray, optional
        CO2 concentration [ppm].
    clip_zero: bool, optional
        if True, replace all negative values with 0.

    Returns
    -------
    pandas.Series or xarray.DataArray containing the calculated potential
    evapotranspiration [W m-2].

    Notes
    -----

    Following :cite:t:`monteith_evaporation_1965`, :cite:t:`allen_crop_1998`,
    :cite:t:`zhang_comparison_2008`, :cite:t:`schymanski_leaf-scale_2017` and
    :cite:t:`yang_hydrologic_2019`.

    .. math:: PET = \\frac{\\Delta (R_{n}-G)+ \\rho_a c_p
        \\frac{e_s-e_a}{r_a}}{\\(\\Delta +\\gamma(1+\\frac{r_s}{r_a}))}

    , where

    .. math:: r_s = f_{co2} * r_l / LAI_{eff}

    .. math:: f_{co2} = (1+S_{r_s}*(CO_2-300))

    ra_method == 0:

    .. math:: r_a = \\frac{208}{u_2}

    ra_method == 1:

    .. math:: r_a = log(\\frac{(zw - d)}{zom}) *
        \\frac{log(\\frac{(zh - d)}{zoh})}{(0.41^2)u_2}

    """
    gamma = calc_psy(pressure)
    dlt = calc_vpc(tmean)
    res_a = calc_res_aero(wind=wind,ra_method=ra_method, croph=croph, zw=zm, zh=zm)
    #res_s = calc_res_surf(
    #    lai=lai, r_s=r_s, r_l=r_l_min, lai_eff=lai_eff, srs=srs, co2=co2, croph=croph
    #)
    res_s = r_l_min   # <-----Raghav (for PET)
    gamma1 = gamma * a_sh / a_s * (1 + res_s / res_a)
    es = calc_e0(tmean) # kPa
    ea = es - vpd   # kPa
    rho_a = calc_rho(pressure, tmean, ea)

    den = (dlt + gamma1)  # Denominator
    
    num1 = dlt * (rn - g) / den
    CP = 1013  # J kg-1 degC-1 
    num2 = rho_a * CP * vpd * a_sh / res_a / den
    pet = num1 + num2
    pet = clip_zeros(pet, clip_zero)
    return pet
#-----------------------------------------------------------------------------
def obj_function_PM(r_l_min, tmean, wind, rn, g, vpd, pressure, zm, lai, croph, co2, obs_PET):
    # Calculate simulated PET using the Penman-Monteith method with the current r_l_min
    sim_PET = calc_pet_penman_monteith(
        tmean=tmean, wind=wind, rn=rn, g=g, vpd=vpd, pressure=pressure, zm=zm, lai=lai, croph=croph, r_l_min=r_l_min,
        r_s=None, ra_method=1, a_sh=1, a_s=1, lai_eff=0, srs=0.0009, 
        co2=co2, clip_zero=True
    )
    error = calc_error(obs_PET, sim_PET)
    return error
#-----------------------------------------------------------------------------
def clip_zeros(s, clip_zero):
    """Method to replace negative values with 0 for Pandas.Series and xarray.DataArray."""
    if clip_zero:
        s = s.where((s >= 0) | s.isnull(), 0)
    return s

def calc_error(obs, pred):
    return np.sqrt(np.mean((obs - pred) ** 2))
def linear_r2_score(X, y):
    """Calculate the R2 score for a linear fit given the feature matrix X and target variable y."""
    mask = np.isfinite(X) & np.isfinite(y)
    X = X[mask]; y = y[mask]
    # Convert X and y to numpy arrays if they are pandas Series
    if isinstance(X, pd.Series):
        X = X.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    
    # Reshape X and y to 2D arrays if they are 1D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    # Create a linear regression model
    model = LinearRegression()
    # Fit the model to the data
    model.fit(X, y)
    # Predict the target variable
    y_pred = model.predict(X)
    # Calculate the R2 score
    r2 = r2_score(y, y_pred)
    return r2


def calc_daily_Rn_raghav(Ta_mean, Ta_max, Ta_min, SRad, VPD, LAI, Lat, Elev, alpha_m, Cr, dates):
    """
    Calculate daily net radiation (Rn) in W m-2 
    
    Parameters:
        Ta_mean : array-like, mean daily air temperature [degC]
        Ta_max  : array-like, max daily air temperature [degC]
        Ta_min  : array-like, min daily air temperature [degC]
        SRad    : array-like, daily incoming shortwave radiation [W m-2]
        VPD     : array-like, vapor pressure deficit [kPa]
        LAI     : array-like, leaf area index
        Lat     : float, site latitude in degrees
        Elev    : float, elevation in meters
        alpha_m : float, max albedo
        Cr      : float, extinction coefficient
        dates   : pandas.DatetimeIndex, date values
        
    Returns:
        Rn : array, net radiation [W m-2]
    """
    sigma = 4.903e-9  # Stefan–Boltzmann constant, MJ K-4 m-2 day-1
    G_sc = 0.082      # Solar constant, MJ m-2 min-1
    Lat_rad = np.radians(Lat)
    
    SRad = SRad*(24*3600) / 1e6 # W m-2 to MJ m-2 day-1
    
    DOY = dates.dayofyear.values
    d_r = 1 + 0.033 * np.cos(2 * pi / 365 * DOY)
    delta = 0.409 * np.sin(2 * pi / 365 * DOY - 1.39)

    ws = np.arccos(np.clip(-np.tan(Lat_rad) * np.tan(delta), -1, 1))

    Ra = (24 * 60 / pi) * G_sc * d_r * (
        ws * np.sin(Lat_rad) * np.sin(delta) +
        np.cos(Lat_rad) * np.cos(delta) * np.sin(ws)
    )

    Rso = (0.75 + 2e-5 * Elev) * Ra
    alpha = alpha_m - (alpha_m - 0.1) * np.exp(-Cr * LAI)
    Rns = (1 - alpha) * SRad

    es_max = 0.6108 * np.exp((17.27 * Ta_max) / (Ta_max + 237.3))
    es_min = 0.6108 * np.exp((17.27 * Ta_min) / (Ta_min + 237.3))
    es = (es_max + es_min) / 2
    ea = np.clip(es - VPD, a_min=0, a_max=None)

    Tmax_K = Ta_max + 273.16
    Tmin_K = Ta_min + 273.16
    Rnl = sigma * ((Tmax_K ** 4 + Tmin_K ** 4) / 2) * \
          (0.34 - 0.14 * np.sqrt(ea)) * \
          (1.35 * SRad / Rso - 0.35)

    Rnl = np.nan_to_num(Rnl, nan=0.0, posinf=0.0, neginf=0.0)
    Rn = np.clip(Rns - Rnl, a_min=0, a_max=None)
    
    return Rn*1e6/(24*3600)
    

def solar_declination(j):
    """Solar declination from day of year [rad].

    Parameters
    ----------
    j: array_like
        day of the year (1-365).

    Returns
    -------
    array_like of solar declination [rad].

    Notes
    -------
    Based on equations 24 in :cite:t:`allen_crop_1998`.

    """
    return 0.409 * sin(2.0 * pi / 365.0 * j - 1.39)


def sunset_angle(sol_dec, lat):
    """Sunset hour angle from latitude and solar declination - daily [rad].

    Parameters
    ----------
    sol_dec: array_like
        solar declination [rad].
    lat: array_like
        the site latitude [rad].

    Returns
    -------
    array_like containing the calculated sunset hour angle - daily [rad].

    Notes
    -----
    Based on equations 25 in :cite:t:`allen_crop_1998`.

    """
    if isinstance(lat, DataArray):
        lat = lat.expand_dims(dim={"time": sol_dec.index}, axis=0)
        return arccos(clip(-tan(sol_dec.values) * tan(lat).T, -1, 1)).T
    else:
        return arccos(clip(-tan(sol_dec) * tan(lat), -1, 1))


def daylight_hours(tindex, lat):
    """Daylight hours [hour].

    Parameters
    ----------
    tindex: pandas.DatetimeIndex
    lat: array_like
        the site latitude [rad].

    Returns
    -------
    pandas.Series or xarray.DataArray containing the calculated daylight hours [hour].

    Notes
    -----
    Based on equation 34 in :cite:t:`allen_crop_1998`.

    """
    j = day_of_year(tindex)
    sol_dec = solar_declination(j)
    sangle = sunset_angle(sol_dec, lat)
    # Account for subpolar belt which returns NaN values
    dl = 24 / pi * sangle
    if isinstance(lat, DataArray):
        sol_dec = ((dl / dl).T * sol_dec.values).T
    dl = where((sol_dec > 0) & (isnan(dl)), nanmax(dl), dl)
    dl = where((sol_dec < 0) & (isnan(dl)), 0, dl)
    return dl


def relative_distance(j):
    """Inverse relative distance between earth and sun from day of the year.

    Parameters
    ----------
    j: array_like
        day of the year (1-365).

    Returns
    -------
    pandas.Series specifying relative distance between earth and sun.

    Notes
    -------
    Based on equations 23 in :cite:t:`allen_crop_1998`.

    """
    return 1 + 0.033 * cos(2.0 * pi / 365.0 * j)


def extraterrestrial_r(tindex, lat):
    """
    Extraterrestrial daily radiation [MJ m-2 d-1].

    Parameters
    ----------
    tindex: pandas.DatetimeIndex
    lat: array_like
        the site latitude [rad].

    Returns
    -------
    array_like containing the calculated extraterrestrial radiation [MJ m-2 d-1]

    Notes
    -----
    Based on equation 21 in :cite:t:`allen_crop_1998`.

    """
    j = day_of_year(tindex)
    dr = relative_distance(j)
    sol_dec = solar_declination(j)

    omega = sunset_angle(sol_dec, lat)
    if isinstance(lat, DataArray):
        lat = lat.expand_dims(dim={"time": sol_dec.index}, axis=0)
        xx = sin(sol_dec.values) * sin(lat.T)
        yy = cos(sol_dec.values) * cos(lat.T)
        return (118.08 / 3.141592654 * dr.values * (omega.T * xx + yy * sin(omega.T))).T
    else:
        xx = sin(sol_dec) * sin(lat)
        yy = cos(sol_dec) * cos(lat)
        return 118.08 / 3.141592654 * dr * (omega * xx + yy * sin(omega))


def calc_res_surf(
    lai=None, r_s=None, srs=0.002, co2=300, r_l=100, lai_eff=0, croph=0.12
):
    """Surface resistance [s m-1].

    Parameters
    ----------
    lai: float or pandas.Series or xarray.DataArray, optional
        leaf area index [-].
    r_s: float or pandas.Series or xarray.DataArray, optional
        surface resistance [s m-1].
    r_l: float or pandas.Series or xarray.DataArray, optional
        bulk stomatal resistance [s m-1].
    lai_eff: float, optional
        1 => LAI_eff = 0.5 * LAI
        2 => LAI_eff = lai / (0.3 * lai + 1.2)
        3 => LAI_eff = 0.5 * LAI; (LAI>4=4)
        4 => see :cite:t:`zhang_comparison_2008`.
    srs: float or pandas.Series or xarray.DataArray, optional
        Relative sensitivity of rl to ?[CO2] :cite:t:`yang_hydrologic_2019`.
    co2: float or pandas.Series or xarray.DataArray
        CO2 concentration [ppm].
    croph: float or pandas.Series or xarray.DataArray, optional crop height [m].

    Returns
    -------
    float or pandas.Series or xarray.DataArray containing the calculated surface
    resistance [s / m]

    """
    if r_s:
        return r_s
    else:
        fco2 = 1 + srs * (co2 - 300)
        if lai is None:
            return fco2 * r_l / (0.5 * croph * 24)  # after FAO-56
        else:
            return fco2 * r_l / calc_laieff(lai=lai, lai_eff=lai_eff)


def calc_laieff(lai=None, lai_eff=0):
    """Effective leaf area index [-].

    Parameters
    ----------
    lai: pandas.Series/float, optional
        leaf area index [-].
    lai_eff: float, optional
        0 => LAI_eff = 0.5 * LAI
        1 => LAI_eff = lai / (0.3 * lai + 1.2)
        2 => LAI_eff = 0.5 * LAI; (LAI>4=4)
        3 => see :cite:t:`zhang_comparison_2008`.

    Returns
    -------
    pandas.Series containing the calculated effective leaf area index.

    """
    if lai_eff == 0:
        return 0.5 * lai
    if lai_eff == 1:
        return lai / (0.3 * lai + 1.2)
    if lai_eff == 2:
        laie = lai.copy()
        laie[(lai > 2) & (lai < 4)] = 2
        laie[lai > 4] = 0.5 * lai
        return laie
    if lai_eff == 3:
        laie = lai.copy()
        laie[lai > 4] = 4
        return laie * 0.5


def calc_res_aero(wind,USTAR=None,croph=0.12, zw=2, zh=2, ra_method=0):
    """Aerodynamic resistance [s m-1].

    Parameters
    ----------
    wind: float or pandas.Series or xarray.DataArray
        mean day wind speed [m/s].
    croph: float or pandas.Series or xarray.DataArray, optional
        crop height [m].
    zw: float, optional
        height of wind measurement [m].
    zh: float, optional
         height of humidity and or air temperature measurement [m].
    ra_method: float, optional
        0 => ra = 208/wind
        1 => ra is calculated based on equation 36 in FAO (1990), ANNEX V.
        2 => ra is calculated based on equation 9 in https://iopscience.iop.org/article/10.1088/1748-9326/ae33d3/pdf

    Returns
    -------
    pandas.Series containing the calculated aerodynamic resistance.

    """
    if ra_method == 0:
        wind = wind.where(wind != 0, 0.0001)
        ra = 208 / wind
        ra = ra.clip(lower=10, upper=5000)
        return ra
    elif ra_method == 2:
        wind = wind.where(wind != 0, 0.0001)
        ra = wind / USTAR**2 + 6.2*USTAR**(-0.67)
        ra = ra.clip(lower=10, upper=5000)
        return ra
    else:
        d = 0.667 * croph
        zom = 0.123 * croph
        zoh = 0.0123 * croph
        ra = (log((zw - d) / zom)) * (log((zh - d) / zoh) / (0.41**2) / wind)
        ra = ra.clip(lower=10, upper=5000)
        return ra
