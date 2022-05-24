#efficiency.py
import xarray as xr
import numpy as np
import datetime as dt
from datetime import datetime
import pandas as pd
import scipy.integrate as integ
import scipy.stats as sp
from scipy import signal

#constants
g = 9.81
Lv = 2.5e6
Cp = 1004.
sid = 86400.

C = 40075000. * np.cos(np.deg2rad(70.))
A = (2.*6371000.**2*np.pi)*(np.sin(np.deg2rad(90.))-np.sin(np.deg2rad(70.))) 

def ufunc(array,function,core_dim=None):
    if core_dim==None:
        return xr.apply_ufunc(
            function, array,
            dask='allowed',
            output_dtypes=[float])
    else:
        return xr.apply_ufunc(
            function, array,
            input_core_dims=[[reduce_dim]],
            exclude_dims={reduce_dim},
            dask='allowed',
            output_dtypes=[float])

def covariance_gufunc(x, y):
    return ((x - x.mean(axis=-1, keepdims=True))
            * (y - y.mean(axis=-1, keepdims=True))).mean(axis=-1)

def pearson_correlation_gufunc(x, y):
    return covariance_gufunc(x, y) / (x.std(axis=-1) * y.std(axis=-1))

def pearson_correlation(x, y, dim):
    return xr.apply_ufunc(
        pearson_correlation_gufunc, x, y,
        input_core_dims=[[dim], [dim]],
        dask='allowed',
        output_dtypes=[float])

def h(circ=C,area=A):
    return (circ/(area*g))*10**4

def anom_daily(array):
    mean = array.groupby('time.dayofyear').mean('time')
    anom = array.groupby('time.dayofyear') - mean
    return anom

def average(ds):
    lon=ds.lon
    lat=ds.lat
    lat_field = xr.broadcast(lat,lon)[0]
    lat_field = (lat_field*(ds.isnull()==False)).where((lat_field*(ds.isnull()==False))>0)
    weights=np.cos(np.deg2rad(lat_field))/np.cos(np.deg2rad(lat_field)).mean('lat')
    avg = (ds*weights).mean(['lon','lat'])
    return avg

def integrate(array):
    """Computes the vertical integral (trapezoidal Rule)."""
    return (array.integrate('lev')/g)*-100

def integrate_trop(array):
    """Computes the vertical integral over the troposphere (1000-300 hPa)"""
    return (array.sel(lev=slice(1000,300)).integrate('lev')/g)*-100

def integrate_strat(array):
    """Computes the vertical integral over the stratosphere (1000-300 hPa)"""
    return (array.sel(lev=slice(300,.1)).integrate('lev')/g)*-100

#alternate integration functions
lev = np.array([1.00000000e+05, 9.75000000e+04, 9.50000000e+04, 9.25000000e+04,
       9.00000000e+04, 8.75000000e+04, 8.50000000e+04, 8.25000000e+04,
       8.00000000e+04, 7.75000000e+04, 7.50000000e+04, 7.25000000e+04,
       7.00000000e+04, 6.50000000e+04, 6.00000000e+04, 5.50000000e+04,
       5.00000000e+04, 4.50000000e+04, 4.00000000e+04, 3.50000000e+04,
       3.00000000e+04, 2.50000000e+04, 2.00000000e+04, 1.50000000e+04,
       1.00000000e+04, 7.00000000e+03, 5.00000000e+03, 4.00000000e+03,
       3.00000000e+03, 2.00000000e+03, 1.00000000e+03, 7.00000000e+02,
       5.00000000e+02, 4.00000000e+02, 3.00000000e+02, 2.00000000e+02,
       1.00000000e+02, 6.99999988e+01, 5.00000000e+01, 4.00000006e+01,
       3.00000012e+01, 1.00000001e+01])
def integrate2(array):
    return (np.squeeze(integ.trapz(array,x=-(lev)/g,axis=-1)))
def integrate_trop2(array):
    return (np.squeeze(integ.trapz(array[:,0:21],x=-(lev[0:21])/g,axis=-1)))
def integrate_strat2(array):
    return (np.squeeze(integ.trapz(array[:,20:42],x=-(lev[20:42])/g,axis=-1)))
def detrend(array):
    return signal.detrend(array,0)

def detrend_rm(array,days=213*5):
    """Alternate detrending method that removes the 5-year moving average over winter"""
    return array - array.rolling(time=days,center=True,min_periods=1).mean()

def low_filt(data,N=1,Wn=.5,btype='low'):
    sos = signal.butter(N, Wn, btype, output='sos')
    filt = signal.sosfiltfilt(sos, data)
    return filt

def composite(array,events):
    comp = xr.concat([array.sel(time=slice(array.time.shift(time=30).sel(time=events[x])
                                           ,array.time.shift(time=-30).sel(time=events[x]))).drop('time')
                      for x in range(len(events))],'event').mean('event')
    return comp

def composite_mean(array,events,s=7,e=0,time=False):
    comp = xr.concat([array.sel(time=slice(array.time.shift(time=-s).sel(time=events[x])
                                           ,array.time.shift(time=-e).sel(time=events[x]))).mean('time')
                      for x in range(len(events))],'event')
    if time == True:
        comp = xr.DataArray(comp,coords=[events.time],dims=['time'])
    return comp

def composite_event(array,events):
    comp = xr.concat([array.sel(time=slice(array.time.shift(time=30).sel(time=events[x])
                                           ,array.time.shift(time=-30).sel(time=events[x]))).drop('time')
                      for x in range(len(events))],'event')
    return comp

def event_days(sigma,array):
    SD = array.std('time')
    high_MSE_time = xr.where(array>(SD*sigma)
         ,array,np.nan)
    eff_high_MSE_time = high_MSE_time.dropna('time').time
    return eff_high_MSE_time

def event_average(array,events):
    "Returns a cumulative average and size of each event. The 'event array' should include all consecutive days in each event. The cumulative average of the event will be returned on the start day of the event."
    df = (xr.broadcast(array.sel(time=events),array.time)[0]).to_dataframe('df')
    df2 = (xr.broadcast(array.sel(time=events).where(xr.ufuncs.logical_and(array>=0,array<=1)),array.time)[0]).to_dataframe('df2')
    df3 = (xr.broadcast(array.sel(time=events),array.time)[0]).to_dataframe('df3')
    df_avg = df2['df2'].groupby(df['df'].isna().cumsum()).expanding().mean().reset_index()
    df_avg2 = df_avg.groupby('df').first().dropna()
    df_avg3 = df_avg.groupby('df').last().dropna()
    bin_size = df3['df3'].groupby(df['df'].isna().cumsum()).expanding().count().reset_index()
    bin_size2 = bin_size.replace(0,np.nan)
    bin_size3 = bin_size2.groupby('df').first().dropna()
    bin_size4 = bin_size2.groupby('df').last().dropna()
    
    start = np.sort(np.concatenate([df_avg2.where(np.logical_and(df_avg2.time.dt.day==31,df_avg2.time.dt.month==3)).dropna().time.to_numpy()+np.array(215, dtype='timedelta64[D]'),
              df_avg2.where(np.logical_or(df_avg2.time.dt.day!=31,df_avg2.time.dt.month!=3)).dropna().time.to_numpy()
             +np.array(1, dtype='timedelta64[D]')]))
    start2 = np.sort(np.concatenate([bin_size3.where(np.logical_and(bin_size3.time.dt.day==31,bin_size3.time.dt.month==3)).dropna().time.to_numpy()+np.array(215, dtype='timedelta64[D]'),
              bin_size3.where(np.logical_or(bin_size3.time.dt.day!=31,bin_size3.time.dt.month!=3)).dropna().time.to_numpy()
             +np.array(1, dtype='timedelta64[D]')]))
    
    end = df_avg3['df2']
    end2 = bin_size4['df3']
    
    c_size = xr.DataArray(end2.values,coords=[start2],dims=['time'])
    c_avg = xr.DataArray(df_avg['df2'].values,coords=[df_avg['time'].values],dims=['time'])
    return c_avg,c_size


def event_integ(array,events):
    "Returns a cumulative integral of each event. The 'event array' should include all consecutive days in each event. The cumulative average of the event will be returned on the start day of the event."
    df = (xr.broadcast(array.sel(time=events),array.time)[0]).to_dataframe('df')
    df2 = (xr.broadcast(array.sel(time=events),array.time)[0]).to_dataframe('df2')
    df_avg = (df2['df2'].fillna(0).groupby(df['df'].isna().cumsum()).expanding()
    .apply(integ.trapz,kwargs={'dx':sid}).replace(0,np.nan).reset_index())
    
    events = pd.concat([df_avg.where(df_avg['df2'] >x*10**6).groupby('df').first() 
                for x in np.arange(8,112,8)]).sort_values(by=['time'])
    c_integ = xr.DataArray(events['df2'].values*10**-6,coords=[events.time],dims=['time'])
    c_integ_full = xr.DataArray(df_avg['df2'].values*10**-6,coords=[df_avg['time'].values],dims=['time'])
    return c_integ,c_integ_full

#alternate composite function
#def composite2(array,events):
#    comp = xr.concat([array.sel(time=slice(events[x]-np.array(30, dtype='timedelta64[D]')
#                                                      ,events[x]+np.array(30, dtype='timedelta64[D]'))).drop('time')
#                      for x in range(len(events))],'event').mean('event')
#    return comp

#alternate method for ttests
#def ttest_time(array,array2,lon=True):
#    if lon==True:
#        array = average(array)
#        array2 = average(array2)
#    return sp.ttest_ind(array,array2).pvalue

#def ttest_lev(array,array2,lon=False):
#    return np.array([sp.ttest_ind(array.isel(lev=x),
#            array2.isel(lev=x)).pvalue for x in range(21)])
