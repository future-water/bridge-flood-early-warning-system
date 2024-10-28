import numpy as np
import pandas as pd
from _crps import crps_ensemble

def brier_score(forecast_probs, actual_outcomes):
    N = len(forecast_probs)
    return sum([(forecast - actual)**2 for forecast, actual in zip(forecast_probs, actual_outcomes)]) / N

def NSE(predicted, observed):
    mean_observed = np.mean(observed)
    numerator = np.sum((observed - predicted) ** 2)
    denominator = np.sum((observed - mean_observed) ** 2)
    return 1 - (numerator / denominator) if denominator != 0 else float('inf')

def calculate_rmse(predictions, observations):
    valid_data = pd.DataFrame({'predictions': predictions, 'observations': observations}).dropna()
    return np.sqrt(((valid_data['predictions'] - valid_data['observations']) ** 2).mean())

def calculate_correlation(predictions, observations):
    valid_data = pd.DataFrame({'predictions': predictions, 'observations': observations}).dropna()
    return valid_data['predictions'].corr(valid_data['observations'])

def load_and_prepare_usgs(file_path, gauges_usgs):
    usgs = pd.read_csv(file_path)
    usgs.set_index('datetime', inplace=True)
    usgs.index = pd.to_datetime(usgs.index).tz_convert('UTC')

    rename_dict = {v.split('-')[-1]: k for k, v in gauges_usgs.items() if v.split('-')[-1] in usgs.columns}
    usgs = usgs.rename(columns=rename_dict)
    
    return usgs, rename_dict

def sensitivity(site, factor, raw, all_outputs_open, all_outputs_da, usgs, date_range, multi_models):
    obs_cms = usgs[site]
    overtop_flow = max(obs_cms) * factor
    # print(f"Overtop flow threshold: {overtop_flow}")

    raw_flow = {}
    for date in date_range:
        raw_flow[date] = raw[date][site].iloc[:19]
            
    site_output = {}
    for date in date_range:
        site_output[date] = all_outputs_open[date][site]
        
    site_output_da = {}
    for date in date_range:
        site_output_da[date] = {}
        for model in multi_models:
            site_output_da[date][model] = all_outputs_da[model][date][site]

    raw_spreads = {}
    raw_spread = pd.concat((value for value in raw_flow.values()), axis=1)
    raw_spread.columns = (pd.to_datetime(key, format='%Y%m%d%H').tz_localize('UTC') for key in raw_flow.keys())
    raw_spreads = raw_spread

    open_spreads = {}
    open_spread = pd.concat((value for value in site_output.values()), axis=1)
    open_spread.columns = (pd.to_datetime(key, format='%Y%m%d%H').tz_localize('UTC') for key in site_output.keys())
    open_spreads = open_spread

    ensemble_weights_history = {}
    median_kde = {}
    overtop_prob_raw = {}
    overtop_prob_da = {}
    overtop_prob_open = {}

    crps_open = {i: [] for i in range(19)}
    crps_raw = {i: [] for i in range(19)}
    crps_da = {i: [] for i in range(19)}

    for date_index, date in enumerate(date_range):
        median_kde[date] = {}
        time_range = site_output_da[date][multi_models[0]].index
        #######################DA############################
        medians_da = {}

        ensemble_weights = np.ones(len(multi_models))
            
        if date_index > 0:
            for i, model in enumerate(multi_models):
                ground_truth = obs_cms[site_output_da[date_range[date_index - 1]][model].index[1]]
                overtop_flow_values = np.array([site_output_da[date_range[date_index - 1]][model].iloc[1]])
                combined_values = overtop_flow_values

                for n in range(2, 19):
                    if date_index >= n:
                        additional_point = np.array([site_output_da[date_range[date_index - n]][model].iloc[n]])
                        combined_values = np.concatenate((combined_values, additional_point))

                mse = np.mean((combined_values - ground_truth)**2)
                ensemble_weights[i] = 1 / (mse + 1e-6)
            
        ensemble_weights /= np.sum(ensemble_weights)
        ensemble_weights_history[date] = ensemble_weights.copy()

        for i, timestamp in enumerate(time_range):
            overtop_flow_values = np.array([site_output_da[date][model][timestamp] for model in multi_models])
            overtop_flow_values += np.random.normal(0, 0.001, size=overtop_flow_values.shape)
            median = np.sum(overtop_flow_values * ensemble_weights)
            medians_da[timestamp] = median

        medians_da = pd.Series(medians_da)
    
        median_kde[date] = medians_da
        obs_all = obs_cms
        overflow_occured_all = pd.Series(np.where(obs_all > overtop_flow, 1., 0.), index=obs_all.index)

    da_spreads = {}
    da_spread = pd.concat((value for value in median_kde.values()), axis=1)
    da_spread.columns = (pd.to_datetime(key, format='%Y%m%d%H').tz_localize('UTC') for key in site_output_da.keys())
    da_spreads = da_spread

    timelagged_das = {}
    timelagged_raws = {}
    timelagged_opens = {}

    timelagged_da_probs = {}
    timelagged_raw_probs = {}
    timelagged_open_probs = {}

    brier_raws = {}
    brier_das = {}
    brier_opens = {}

    crps_raws = {}
    crps_das = {}
    crps_opens = {}

    for hour in range(1, 13):

        def mean_first_12_hours(row):
            valid_values = row.dropna()[-7-hour:-hour]
            return valid_values.mean()

        def prob_first_12_hours(row):
            valid_values = row.dropna()[-7-hour:-hour]
            prob = np.sum(np.where(valid_values > overtop_flow, 1., 0.))/len(valid_values)
            return prob

        def crps_first_12_hours(row):
            valid_values = row.dropna()[-7-hour:-hour]
            crps_score = crps_ensemble(obs_cms[valid_values.index], valid_values)
            return crps_score.mean()


        timelagged_da = da_spreads.apply(mean_first_12_hours, axis=1)
        timelagged_raw = raw_spreads.apply(mean_first_12_hours, axis=1)
        timelagged_open = open_spreads.apply(mean_first_12_hours, axis=1)

        crps_da = da_spreads.apply(crps_first_12_hours, axis=1).mean()
        crps_raw = raw_spreads.apply(crps_first_12_hours, axis=1).mean()
        crps_open = open_spreads.apply(crps_first_12_hours, axis=1).mean()
        
        timelagged_da_prob = da_spreads.apply(prob_first_12_hours, axis=1)
        timelagged_raw_prob = raw_spreads.apply(prob_first_12_hours, axis=1)
        timelagged_open_prob = open_spreads.apply(prob_first_12_hours, axis=1)
        
        brier_raw = brier_score(timelagged_raw_prob.dropna(), overflow_occured_all[timelagged_raw_prob.dropna().index])
        brier_da = brier_score(timelagged_da_prob.dropna(), overflow_occured_all[timelagged_da_prob.dropna().index])
        brier_open = brier_score(timelagged_open_prob.dropna(), overflow_occured_all[timelagged_open_prob.dropna().index])

        timelagged_das[hour] = timelagged_da
        timelagged_raws[hour] = timelagged_raw
        timelagged_opens[hour] = timelagged_open

        timelagged_da_probs[hour] = timelagged_da_prob
        timelagged_raw_probs[hour] = timelagged_raw_prob
        timelagged_open_probs[hour] = timelagged_open_prob

        brier_raws[hour] = brier_raw
        brier_das[hour] = brier_da
        brier_opens[hour] = brier_open

        crps_raws[hour] = crps_raw
        crps_das[hour] = crps_da
        crps_opens[hour] = crps_open
    
    return ensemble_weights_history, overtop_flow, timelagged_das, timelagged_raws, timelagged_opens, crps_das, crps_raws, crps_opens, timelagged_da_probs, timelagged_raw_probs, timelagged_open_probs, brier_raws, brier_das, brier_opens  

def calculate_diff(results, gage_list, key_da, key_raw, epsilon=1e-6):
    diff = {}
    diff_mean = np.zeros(len(gage_list))

    for i, site in enumerate(gage_list):
        crps_da = np.array(list(results[i][key_da].values()))
        crps_raw = np.array(list(results[i][key_raw].values()))

        crps_da = np.nan_to_num(crps_da, nan=0)
        crps_raw = np.nan_to_num(crps_raw, nan=0) + epsilon

        diff_values = 1 - crps_da / crps_raw
        diff[site] = diff_values
        diff_mean[i] = np.mean(diff_values)

    return diff, diff_mean

def prepare_data_for_plotting(diff_da, diff_raw):
    bss_da_skill = pd.DataFrame(list(diff_da.values())).T
    bss_raw_skill = pd.DataFrame(list(diff_raw.values())).T

    combined_data = pd.DataFrame({
        'DA Skill': bss_da_skill.stack(),
        'Raw Skill': bss_raw_skill.stack()
    })
    combined_data = combined_data.reset_index()
    combined_data.columns = ['Index', 'SubIndex', 'DA Skill', 'Raw Skill']

    melted_data = combined_data.melt(id_vars=['Index', 'SubIndex'], value_vars=['DA Skill', 'Raw Skill'], 
                                     var_name='Skill Type', value_name='Skill Value')
    melted_data['Index'] = melted_data['Index'] + 1
    return melted_data

def calculate_ensemble_weights_and_flows(date_range, site_output_da, obs_cms, multi_models):
    ensemble_weights_history = {}
    weighted_flows = {}

    for date_index, date in enumerate(date_range):
        weighted_flows[date] = {}
        time_range = site_output_da[date][multi_models[0]].index

        weighted_flow = {}
        ensemble_weights = np.ones(len(multi_models))

        if date_index > 0:
            for i, model in enumerate(multi_models):
                ground_truth = obs_cms[site_output_da[date_range[date_index - 1]][model].index[1]]
                overtop_flow_values = np.array([site_output_da[date_range[date_index - 1]][model].iloc[1]])
                combined_values = overtop_flow_values

                for n in range(2, 19):
                    if date_index >= n:
                        additional_point = np.array([site_output_da[date_range[date_index - n]][model].iloc[n]])
                        combined_values = np.concatenate((combined_values, additional_point))

                mse = np.mean((combined_values - ground_truth)**2)
                ensemble_weights[i] = 1 / (mse + 1e-6)

        ensemble_weights /= np.sum(ensemble_weights)
        ensemble_weights_history[date] = ensemble_weights.copy()

        for timestamp in time_range:
            overtop_flow_values = np.array([site_output_da[date][model][timestamp] for model in multi_models])
            weighted_flow[timestamp] = np.sum(overtop_flow_values * ensemble_weights)

        weighted_flows[date] = pd.Series(weighted_flow)

    return ensemble_weights_history, weighted_flows