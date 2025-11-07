
import pandas as pd
import numpy as np
from shapely.geometry import box
from skysurvey import Survey
from skysurvey.tools.utils import get_skynoise_from_maglimit
from skysurvey.target import SNeIa
from skysurvey.target import core
from shapely.ops import unary_union
from skysurvey.effects import mw_extinction
import matplotlib.pyplot as plt
from skysurvey import dataset
from shapely import geometry
import matplotlib.pyplot as plt
from astropy.time import Time
from datetime import datetime
from skysurvey.target import SNeII
from astropy.coordinates import SkyCoord
import astropy.units as u
import json
import requests
import os
import dustmaps.planck
dustmaps.planck.fetch()

import warnings
warnings.filterwarnings('ignore')

debug = False

## Check if obs.json exists
if not os.path.exists("obs.json"):
    fritz_secret = 'secret.txt'
    try:
        with open(fritz_secret, 'r') as file:
            fritz_token = file.read()
    except FileNotFoundError:
        print(f"File {fritz_secret} not found.")

    url = f"https://fritz.science/api/observation"

    querystring = {"startDate":"2021-06-01","endDate":"2025-07-30","numPerPage":10000}

    headers = {"Authorization": f'token {fritz_token}'}

    response = requests.get(url, headers=headers, params=querystring)

    a = response.json()

    with open("obs.json",'w') as f:
        f.write(json.dumps(a, indent=4))
else:
    print('using existing obs.json file')
    # Read existing JSON file
    with open("obs.json", 'r') as f:
        a = json.load(f)

mjd = []
dt = []
filter = []
limmag = []
ra = []
dec = []

for i in a["data"]["observations"]:
    dt.append(i["obstime"])
    dt_object = datetime.fromisoformat(i["obstime"])
    t = Time(dt_object, format='datetime')
    mjd.append(t.mjd)
    ra.append(i["field"]["ra"])
    dec.append(i["field"]["dec"])
    filter.append(i["filt"])
    limmag.append(i["limmag"])
    
data = np.array([mjd, filter, limmag, ra, dec])
df = pd.DataFrame(data.T, columns=["mjd", "band", "limmag", "ra", "dec"])
df['limmag']=df.limmag.astype('float32')
df['ra']=df.ra.astype('float32')
df['dec']=df.dec.astype('float32')
df['mjd']=df.mjd.astype('float32')
print(df.dtypes)
df["zp"] = 30
df["gain"] = 6.0
df["limmag"].astype("Float32")
df["skynoise"] = df["limmag"].apply(get_skynoise_from_maglimit, zp=30).values
#df = df[df['band'] != 'ztfg']


coords = ((0., 0.), (0., 7.), (7., 7.), (7., 0.), (0., 0.))
footprint = geometry.Polygon(coords)
dataset_bool = False
attempt_count = 1
while not dataset_bool:
    print('starting dataset creation attempt ', attempt_count)  if debug else None
    try:
        mysurvey = Survey.from_pointings(df, footprint=footprint)
        print('made it past mysurvey') if debug else None
        #mysurvey.show()

        fov_deg = 40.0 / 60.0
        half_size = fov_deg / 2.0
        footprint = box(-half_size, -half_size, half_size, half_size)
        print('defined footprint') if debug else None

        tiles = []
        for _, row in df.iterrows():
            ra = row["ra"]
            dec = row["dec"]
            tile = box(ra - half_size, dec - half_size, ra + half_size, dec + half_size)
            tiles.append(tile)
        tstart, tstop = mysurvey.get_timerange()
        skyarea = unary_union(tiles)
        print('defined tiles and skyarea') if debug else None

        sniia = SNeII.from_draw(
            tstart=tstart,
            tstop=tstop,
            skyarea=skyarea,
            zmin=0.0,
            zmax=2,  #Note : redshift changed to 2
            effect=mw_extinction,
            rate= 1e5/5,
            template = ['v19-2016gkg-corr', 'v19-2011ei-corr']
            )
        print('defined sniia') if debug else None
        ## where issue lies
        dset = dataset.DataSet.from_targets_and_survey(sniia, mysurvey)
        print('created dataset') if debug else None
        dataset_bool = True
    except Exception as e:
        print(f"An error occurred: {e}")
        attempt_count += 1
        print(f"Retrying dataset creation... (Attempt {attempt_count})")

def change_dataset_entry(dset):
    idx =  dset.obs_index.to_numpy()
    list_df_all = []

    for index in idx:
        new_data = dset.data.xs(index).copy()
        new_data['mjd_int'] = new_data['mjd'].astype(int)
        lc_band = new_data.groupby('band')
        list_df = []
        for band, df_band in lc_band:
            lc_band_g = df_band.groupby('mjd_int')
            for mjd, sub_df in lc_band_g:
                sub_df['w'] = 1/sub_df['fluxerr']
                sub_df['weighted_f'] = sub_df['flux']*sub_df['w']/(sub_df['w'].sum())
                sub_df['weighted_fluxerr'] = sub_df['fluxerr']*sub_df['w']/(sub_df['w'].sum())
                sub_df['mjd'] = mjd
                dict={
                    'index_obs': [sub_df.index[0]],
                    'fieldid': [sub_df['fieldid'].values[0]],
                    'band': [sub_df['band'].values[0]],
                    'mjd' : [mjd],  
                    'zp'  : [30],
                    'zpsys' : ['ab'],
                    'gain'  : [1.0],
                    'skynoise' : [sub_df['skynoise'].mean()],
                    'flux' : [sub_df['weighted_f'].sum()],    
                    'fluxerr' : [sub_df['weighted_fluxerr'].sum()]}
            
                list_df.append( pd.DataFrame.from_dict(dict))
                                        

        df = pd.concat(list_df)
        df.reset_index(drop=True,inplace=True)

        index_tuples = [(index, a) for a in df.index_obs]
        multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['index', 'index_obs'])
        df.drop('index_obs',axis=1, inplace=True)
        df = df.set_index(multi_index)
        list_df_all.append(df)

    final_df = pd.concat(list_df_all)
    return final_df


# For each light curve i compute the SNR and then if any entry in any band 
# satisfies the threshold, i count them as detections 
# This can be further changes depending on weather we want more cuts
def num_of_detections_new(df, SNR):
    indices = np.unique(df.index.get_level_values(0).to_numpy())
    valid_light_curves = []
    valid_SNR_index = []
    for idx in indices:
        list_df = []
        lc = df.xs(idx)
        SNR_vals = lc['flux']/lc['fluxerr']
        if (SNR_vals > SNR).any() and (len(lc)>=7):
            valid_SNR_index.append(idx)
            valid_light_curves.append(lc)

    print(valid_SNR_index)
    print(f"Number of valid detections with SNR > {SNR} is {len(valid_SNR_index)}")
    return valid_SNR_index, valid_light_curves

df = change_dataset_entry(dset)
valid_idx, valid_lightcurves = num_of_detections_new(df, 5)

def plot_lightcurve(i, df):
    coef = 10 ** (-(df["zp"] - 25) / 2.5)
    df["flux_zp"] = df["flux"] * coef
    df["fluxerr_zp"] = df["fluxerr"] * coef

    from matplotlib import dates as mdates        
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    fig, ax = plt.subplots()

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    grouped_df = df.groupby('band')
    color = {
        'ztfr' : 'red',
        'ztfg' : 'green',
        'ztfi' : 'indigo',
    }
    for group, band_df in grouped_df:
        times = Time(band_df["mjd"], format="mjd").datetime
        ax.scatter(times, band_df['flux_zp'],c=color[group],label=group)
        ax.errorbar(times, band_df["flux_zp"],
            yerr= band_df["fluxerr_zp"],
            ls="None", marker="None", ecolor="grey", 
            zorder=3)
        ax.set_ylabel("Flux[zp=25]")
        ax.grid()
        plt.legend()

        plt.savefig(f"figs/LC_{i}.png")

for i, index in enumerate(valid_idx):
    plot_lightcurve(i, valid_lightcurves[i])