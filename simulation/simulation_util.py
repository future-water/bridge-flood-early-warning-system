import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from muskingum import MuskingumCunge, KalmanFilter

def read_csv(path):
    dataframe = pd.read_csv(path)
    dataframe['time'] = pd.to_datetime(dataframe['time'])
    dataframe = dataframe.set_index('time').tz_localize('UTC')
    return dataframe

def load_nwm_data(basepath, nwm_export, date, nwm_vars):
    return {nwm_var: read_csv(f'{basepath}/{nwm_export}/{date}/{date}_short_range_{nwm_var}.csv') for nwm_var in nwm_vars}

def prepare_inputs(nwm_dataframes):
    q_L = nwm_dataframes['qSfcLatRunoff'] + nwm_dataframes['qBucket']
    q_L.columns = q_L.columns.astype(str)
    return q_L

def update_model_inputs(model_collection, inputs):
    for key in model_collection.inputs:
        reach_ids = model_collection.inputs[key].columns
        model_collection.inputs[key].index = inputs.index
        model_collection.inputs[key].values[:] = inputs[reach_ids].values

def save_raw_data(date_list, basepath, nwm_export, nwm_vars, mc):
    raw = {}
    for date in tqdm(date_list, desc="Saving raw data"):
        date_str = date.strftime('%Y%m%d%H')
        start_datetime = pd.to_datetime(date_str, format='%Y%m%d%H').tz_localize('UTC')
        nwm_dataframes = load_nwm_data(basepath, nwm_export, date_str, nwm_vars)
        ana_dataframes = {nwm_var: read_csv(f'{basepath}/{nwm_export}/{date_str}/{date_str}_data_assimilation_{nwm_var}.csv') for nwm_var in nwm_vars}

        ana_row = ana_dataframes['streamflow'][mc.reach_ids].loc[start_datetime]
        ana_row_df = pd.DataFrame([ana_row.values], index=[ana_row.name], columns=ana_row.index)
        nwm_df = nwm_dataframes['streamflow'][mc.reach_ids]
        raw[date_str] = pd.concat([ana_row_df, nwm_df]).sort_index()
    return raw

async def simulate_open(date_list, basepath, nwm_export, nwm_vars, mc, model_collection):
    all_outputs_open = {}
    for date in tqdm(date_list, desc="Simulating open model"):
        date_str = date.strftime('%Y%m%d%H')
        model_collection.load_states()
        nwm_dataframes = load_nwm_data(basepath, nwm_export, date_str, nwm_vars)
        inputs = prepare_inputs(nwm_dataframes)
        update_model_inputs(model_collection, inputs)
        multi_outputs = await model_collection.simulate_async(verbose=False)
        multi_outputs = pd.concat([output for output in multi_outputs.values()], axis=1)
        all_outputs_open[date_str] = multi_outputs
    return all_outputs_open

async def simulate_model_with_callbacks(date_list, basepath, nwm_export, nwm_vars, model_collection_da, model_name, alpha, usgs):
    all_outputs_da = {}
    for date in tqdm(date_list, desc=f"Simulating model {model_name}"):
        date_str = date.strftime('%Y%m%d%H')
        model_collection_da[model_name].load_states()
        
        nwm_dataframes = load_nwm_data(basepath, nwm_export, date_str, nwm_vars)
    
        inputs = prepare_inputs(nwm_dataframes) * alpha
        inputs.columns = inputs.columns.astype(str)
    
        update_model_inputs(model_collection_da[model_name], inputs)
    
        for key in model_collection_da[model_name].inputs:
            model = model_collection_da[model_name].models[key]
            if hasattr(model, 'callbacks') and 'kf' in model.callbacks:
                measurement_timestep = date
                measurements_columns = model.callbacks['kf'].measurements.columns
                model.callbacks['kf'].measurements = usgs.loc[[measurement_timestep], measurements_columns]
        
        multi_outputs = await model_collection_da[model_name].simulate_async(verbose=False)
        multi_outputs = pd.concat([output for output in multi_outputs.values()], axis=1)
        all_outputs_da[date_str] = multi_outputs
        
    return all_outputs_da
