import os
import numpy as np
import pandas as pd
import quadprog
from sklearn.metrics import r2_score
import scipy.linalg

def solve_parameters(model, i_hat, o_hat, inputs, datetimes, regularization=1e-12):
    p_t = (inputs.reindex(model.reach_ids, axis=1).fillna(0.).loc[datetimes, :].values)
    dt = model.dt
    m = 1 + 2 * model.n
    o_hat_prev = o_hat[:-1]
    o_hat_next = o_hat[1:]
    i_hat_prev = i_hat[:-1]
    i_hat_next = i_hat[1:]
    T1 = o_hat_next - o_hat_prev
    T2 = i_hat_next - i_hat_prev - o_hat_next + o_hat_prev
    T3 = (0.5 * o_hat_next - 0.5 * i_hat_next - 0.5 * i_hat_prev + 0.5 * o_hat_prev - p_t[1:])
    N = T1.shape[0]
    G = np.zeros((m, m))
    G[0, 0] = (T3**2).sum() / N
    G[1:, 0] = np.column_stack([(T1 * T3).sum(axis=0) / N, (T2 * T3).sum(axis=0) / N]).ravel()
    G[0, 1:] = np.column_stack([(T1 * T3).sum(axis=0) / N, (T2 * T3).sum(axis=0) / N]).ravel()
    blocks = np.column_stack([(T1**2).sum(axis=0) / N, (T1 * T2).sum(axis=0) / N, (T1 * T2).sum(axis=0) / N, (T2**2).sum(axis=0) / N])
    blocks = scipy.linalg.block_diag(*blocks.reshape(-1, 2, 2))
    G[1:, 1:] = blocks

    C_block = np.array([0.5, -1, 0, 1, 1, 0, 1, -1]).reshape(4, 2)
    C = np.zeros((1 + 4*model.n, 1 + 2*model.n))
    C[0,0] = 1.
    C[1:, 1:] = scipy.linalg.block_diag(*[C_block] * model.n)
    b = 0.001 * np.ones(1 + 4 * model.n)
    b[0] = dt
    b[4::4] = dt / 2
    a = np.zeros(m)
    G_reg = G + np.eye(m) * regularization

    result = quadprog.solve_qp(G_reg, a, C.T, b, meq=1)
    x = result[0]
    theta = x[1:].reshape(-1, 2)
    K = theta[:,0]
    X = theta[:,1] / theta[:,0]
    return K, X

def nse(observed, modeled):
    return 1 - np.sum((modeled - observed) ** 2) / np.sum((observed - np.mean(observed)) ** 2)

def kge(observed, modeled):
    r = np.corrcoef(observed, modeled)[0, 1]
    alpha = np.std(modeled) / np.std(observed)
    beta = np.mean(modeled) / np.mean(observed)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

def load_data(basepath, nwm_export, date_list):
    all_inputs = []
    all_streamflows = []

    for date in date_list[:-1]:
        date_str = date.strftime('%Y%m%d%H')
        qBucket_path = f'{basepath}/{nwm_export}/{date_str}/{date_str}_no_data_assimilation_qBucket.csv'
        qSfcLatRunoff_path = f'{basepath}/{nwm_export}/{date_str}/{date_str}_no_data_assimilation_qSfcLatRunoff.csv'
        streamflow_path = f'{basepath}/{nwm_export}/{date_str}/{date_str}_no_data_assimilation_streamflow.csv'

        if os.path.exists(qBucket_path) and os.path.exists(qSfcLatRunoff_path) and os.path.exists(streamflow_path):
            qBucket_da = pd.read_csv(qBucket_path, parse_dates=['time'], index_col='time').tz_localize('UTC')
            qSfcLatRunoff_da = pd.read_csv(qSfcLatRunoff_path, parse_dates=['time'], index_col='time').tz_localize('UTC')
            streamflow_da = pd.read_csv(streamflow_path, parse_dates=['time'], index_col='time').tz_localize('UTC')

            inputs = qSfcLatRunoff_da.add(qBucket_da, fill_value=0)
            all_inputs.append(inputs.loc[date].to_frame().T)
            all_streamflows.append(streamflow_da.loc[date].to_frame().T)

    all_inputs = pd.concat(all_inputs)
    all_streamflows = pd.concat(all_streamflows)

    return all_inputs, all_streamflows