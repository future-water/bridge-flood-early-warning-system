import os
import copy
import asyncio
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import scipy.linalg
from scipy.sparse import lil_matrix, csgraph
from scipy.optimize import root_scalar
import numpy as np
import pandas as pd
import datetime
from numba import njit, prange
import quadprog
from nutils import interpolate_sample

logger = logging.getLogger(__name__)

PROCESSES = mp.cpu_count() - 1
MIN_SLOPE = 1e-8

class MuskingumCunge:
    def __init__(self, json, init_o_t=None, init_P_t=None, dt=3600.0,
                 t_0=0.0, create_state_space=True, sparse=False, verbose=False):
        self.verbose = verbose
        self.sparse = sparse
        self.callbacks = {}
        # Read json input file
        assert isinstance(json, dict)
        self.read_nhd_file(json)
        # Construct network topology
        self.construct_network()
        # Create parameter arrays
        n = self.endnodes.size
        self.n = n
        self.alpha = np.zeros(n, dtype=np.float64)
        self.beta = np.zeros(n, dtype=np.float64)
        self.chi = np.zeros(n, dtype=np.float64)
        self.gamma = np.zeros(n, dtype=np.float64)
        self.h = np.zeros(n, dtype=np.float64)
        self.Qn = np.zeros(n, dtype=np.float64)
        if init_o_t is None:
            self.o_t_prev = 1e-3 * np.ones(n, dtype=np.float64)
            self.o_t_next = 1e-3 * np.ones(n, dtype=np.float64)
            self.i_t_prev = np.zeros(len(self.o_t_prev))
            np.add.at(self.i_t_prev, self.endnodes, self.o_t_prev)
            self.i_t_next = np.zeros(len(self.o_t_next))
            np.add.at(self.i_t_next, self.endnodes, self.o_t_next)
        else:
            assert isinstance(init_o_t, np.ndarray)
            assert init_o_t.ndim == 1
            assert init_o_t.size == n
            self.o_t_prev = init_o_t.copy()
            self.o_t_next = init_o_t.copy()
            self.i_t_prev = np.zeros(len(self.o_t_prev))
            np.add.at(self.i_t_prev, self.endnodes, self.o_t_prev)
            self.i_t_next = np.zeros(len(self.o_t_next))
            np.add.at(self.i_t_next, self.endnodes, self.o_t_next)
        # Initialize state-space matrices
        if sparse:
            A = lil_matrix((n, n))
            B = lil_matrix((n, n))
        else:
            self.A = np.zeros((n, n), dtype=np.float64)
            self.B = np.zeros((n, n), dtype=np.float64)
        # Compute parameters
        timedelta = datetime.timedelta(seconds=dt)
        self.t = t_0
        self.datetime = pd.to_datetime(t_0, unit="s", utc=True)
        self.dt = dt
        self.timedelta = timedelta
        self.saved_states = {}
        self.K = np.zeros(n, dtype=np.float64)
        self.X = np.zeros(n, dtype=np.float64)
        # TODO: Should allow these to be set
        self.K[:] = 3600.
        self.X[:] = 0.29
        self.compute_muskingum_coeffs()
        if create_state_space:
            self.create_state_space()
        self.save_state()

    def read_nhd_file(self, d):
        if self.verbose:
            logger.info('Reading NHD file...')
        node_ids = [i['attributes']['COMID'] for i in d['features']]
        link_ids = [i['attributes']['COMID'] for i in d['features']]
        paths = [np.asarray(i['geometry']['paths']) for i in d['features']]
        reach_ids = link_ids
        reach_ids = [str(x) for x in reach_ids]
        source_node_ids = [i['attributes']['COMID'] for i in d['features']]
        target_node_ids = [i['attributes']['toCOMID'] for i in d['features']]
        dx = np.asarray([i['attributes']['Shape_Length'] for i in d['features']]) #* 1000 # km to m
        self.reach_ids = reach_ids
        self.node_ids = node_ids
        self.source_node_ids = source_node_ids
        self.target_node_ids = target_node_ids
        self.dx = dx
        self.paths = paths

    def read_hydraulic_geometry(self, d):
        raise NotImplementedError
        source_node_ids = self.source_node_ids
        target_node_ids = self.target_node_ids
        # Set trapezoidal geometry
        Bws = []
        h_maxes = []
        zs = []
        for link in d["links"]:
            geom = link["hydrologic_routing"]["muskingum_cunge_station"][
                "cross_section"
            ]
            Tw = geom[3]["lr"] - geom[0]["lr"]
            Bw = geom[2]["lr"] - geom[1]["lr"]
            h_max = geom[0]["z"] - geom[1]["z"]
            z = (geom[1]["lr"] - geom[0]["lr"]) / h_max
            Bws.append(Bw)
            h_maxes.append(h_max)
            zs.append(z)
        self.Bw = np.asarray(Bws)
        self.h_max = np.asarray(h_maxes)
        self.z = np.asarray(zs)
        n = self.Bw.size
        node_elevs = {i["uid"]: i["location"]["z"] for i in d["nodes"]}
        So = []
        for i in range(len(source_node_ids)):
            source_node_id = source_node_ids[i]
            target_node_id = target_node_ids[i]
            z_0 = node_elevs[source_node_id]
            z_1 = node_elevs.get(target_node_id, z_0 - 1)
            dx_i = d["links"][i]["length"]
            slope = (z_0 - z_1) / dx_i
            So.append(max(slope, MIN_SLOPE))
        self.So = np.asarray(So)
        self.Ar = np.zeros(n, dtype=np.float64)
        self.P = np.zeros(n, dtype=np.float64)
        self.R = np.zeros(n, dtype=np.float64)
        self.Tw = np.zeros(n, dtype=np.float64)

    def construct_network(self):
        if self.verbose:
            logger.info('Constructing network...')
        # NOTE: node_ids and source_node_ids should always be the same for NHD
        # But not necessarily so for original json files
        node_ids = self.node_ids
        source_node_ids = self.source_node_ids
        target_node_ids = self.target_node_ids
        self_loops = []
        node_index_map = pd.Series(np.arange(len(self.node_ids)), index=self.node_ids)
        startnodes = node_index_map.reindex(source_node_ids, fill_value=-1).values
        endnodes = node_index_map.reindex(target_node_ids, fill_value=-1).values
        for i in range(len(startnodes)):
            if endnodes[i] == -1:
                self_loops.append(i)
                endnodes[i] = startnodes[i]
        indegree = np.bincount(endnodes.ravel(), minlength=startnodes.size)
        for self_loop in self_loops:
            indegree[self_loop] -= 1
        self.startnodes = startnodes
        self.endnodes = endnodes
        self.indegree = indegree

    def compute_normal_depth(self, Q, mindepth=0.01):
        So = self.So
        Bw = self.Bw
        z = self.z
        mann_n = self.mann_n
        h = []
        for i in range(Q.size):
            Qi = Q[i]
            Bi = Bw[i]
            zi = z[i]
            Soi = So[i]
            mann_ni = mann_n[i]
            result = root_scalar(
                _solve_normal_depth,
                args=(Qi, Bi, zi, mann_ni, Soi),
                x0=1.0,
                fprime=_dQ_dh,
                method="newton",
            )
            h.append(result.root)
        h = np.asarray(h, dtype=np.float64)
        h = np.maximum(h, mindepth)
        self.h[:] = h

    def compute_hydraulic_geometry(self, h):
        Bw = self.Bw
        z = self.z
        So = self.So
        mann_n = self.mann_n
        Ar = (Bw + h * z) * h
        P = Bw + 2 * h * np.sqrt(1.0 + z**2)
        Tw = Bw + 2 * z * h
        R = Ar / P
        R[P <= 0] = 0.0
        Qn = np.sqrt(So) / mann_n * Ar * R ** (2 / 3)
        self.Ar[:] = Ar
        self.P[:] = P
        self.R[:] = R
        self.Tw[:] = Tw
        self.Qn[:] = Qn

    def compute_K_and_X(self, h, Q):
        if self.verbose:
            print('Computing K and X...')
        Bw = self.Bw
        z = self.z
        R = self.R
        Tw = self.Tw
        So = self.So
        Ar = self.Ar
        mann_n = self.mann_n
        dt = self.dt
        dx = self.dx
        K = self.K
        X = self.X
        # c_k = (np.sqrt(So) / mann_n) * ( (5. / 3.) * R**(2. / 3.) -
        #          ((2. / 3.) * R**(5. / 3.) * (2. * np.sqrt(1. + z**2)
        #                                       / (Bw + 2 * h * z))))
        # c_k = 1.67 * Q / Ar
        # c_k = np.maximum(c_k, 0.)
        # K[:] = dx / c_k
        # cond = c_k > 0
        # K[cond] = np.maximum(dt, dx[cond] / c_k[cond])
        # K[~cond] = dt
        X[:] = 0.29
        # X[cond] = np.minimum(0.5, np.maximum(0.,
        #           0.5 * (1 - (Q[cond] / (2. * Tw[cond] * So[cond]
        #                                  * c_k[cond] * dx[cond])))))
        # X[~cond] = 0.5
        self.K = K
        self.X = X

    def compute_alpha(self, K, X, dt):
        alpha = (dt - 2 * K * X) / (2 * K * (1 - X) + dt)
        return alpha

    def compute_beta(self, K, X, dt):
        beta = (dt + 2 * K * X) / (2 * K * (1 - X) + dt)
        return beta

    def compute_chi(self, K, X, dt):
        chi = (2 * K * (1 - X) - dt) / (2 * K * (1 - X) + dt)
        return chi

    def compute_gamma(self, K, X, dt):
        gamma = dt / (K * (1 - X) + dt / 2)
        return gamma

    def compute_muskingum_coeffs(self, K=None, X=None, dt=None):
        if self.verbose:
            print('Computing Muskingum coefficients...')
        if K is None:
            K = self.K
        if X is None:
            X = self.X
        if dt is None:
            dt = self.dt
        self.alpha[:] = self.compute_alpha(K, X, dt)
        self.beta[:] = self.compute_beta(K, X, dt)
        self.chi[:] = self.compute_chi(K, X, dt)
        self.gamma[:] = self.compute_gamma(K, X, dt)

    def create_state_space(self, overwrite_old=True):
        if self.verbose:
            print('Creating state-space system...')
        startnodes = self.startnodes
        endnodes = self.endnodes
        alpha = self.alpha
        beta = self.beta
        chi = self.chi
        gamma = self.gamma
        indegree = self.indegree
        A = self.A
        B = self.B
        if overwrite_old:
            self.A.fill(0.0)
            self.B.fill(0.0)
        startnodes = startnodes[(indegree == 0)]
        A, B = self.muskingum_matrix(
            A, B, startnodes, endnodes, alpha, beta, chi, gamma, indegree
        )
        self.A = A
        self.B = B

    def muskingum_matrix(
        self, A, B, startnodes, endnodes, alpha, beta, chi, gamma, indegree
    ):
        m = startnodes.size
        n = endnodes.size
        indegree_t = indegree.copy()
        for k in range(m):
            startnode = startnodes[k]
            endnode = endnodes[startnode]
            while indegree_t[startnode] == 0:
                alpha_i = alpha[startnode]
                beta_i = beta[startnode]
                chi_i = chi[startnode]
                gamma_i = gamma[startnode]
                alpha_j = alpha[endnode]
                beta_j = beta[endnode]
                A[startnode, startnode] = chi_i
                B[startnode, startnode] = gamma_i
                if startnode != endnode:
                    A[endnode, startnode] += beta_j
                    A[endnode] += alpha_j * A[startnode]
                    B[endnode] += alpha_j * B[startnode]
                indegree_t[endnode] -= 1
                startnode = endnode
                endnode = endnodes[startnode]
        return A, B

    def init_states(self, o_t_next=None, i_t_next=None):
        if o_t_next is None:
            self.o_t_next[:] = 0.
        else:
            self.o_t_next[:] = o_t_next[:]
        if i_t_next is None:
            self.i_t_next[:] = 0.
            np.add.at(self.i_t_next, self.endnodes, self.o_t_next[self.endnodes])
        else:
            self.i_t_next[:] = i_t_next[:]

    def step(self, p_t_next, num_iter=1, inc_t=False):
        raise NotImplementedError
        A = self.A
        B = self.B
        dt = self.dt
        timedelta = self.timedelta
        o_t_prev = self.o_t_next
        o_t_next = A @ o_t_prev + B @ p_t_next
        self.o_t_next = o_t_next
        self.o_t_prev = o_t_prev
        if inc_t:
            self.t += dt
            self.datetime += self.timedelta

    def step_iter(self, p_t_next, timedelta=None, inc_t=True):
        if timedelta is None:
            timedelta = self.timedelta
            dt = self.dt
        else:
            dt = float(timedelta.seconds)
        startnodes = self.startnodes
        endnodes = self.endnodes
        indegree = self.indegree
        sub_startnodes = startnodes[(indegree == 0)]
        o_t_prev = self.o_t_next
        i_t_prev = self.i_t_next
        if dt != self.dt:
            self.compute_muskingum_coeffs(dt=dt)
        alpha = self.alpha
        beta = self.beta
        chi = self.chi
        gamma = self.gamma
        for _, callback in self.callbacks.items():
            callback.__on_step_start__()
        i_t_next, o_t_next = _ax_bu(sub_startnodes, endnodes, alpha, beta, chi, gamma,
                                    i_t_prev, o_t_prev, p_t_next, indegree)
        self.o_t_next = o_t_next
        self.o_t_prev = o_t_prev
        self.i_t_next = i_t_next
        self.i_t_prev = i_t_prev
        if inc_t:
            self.datetime += timedelta
            self.t += dt
        for _, callback in self.callbacks.items():
            callback.__on_step_end__()

    def propagate_uncertainty(self, Q_cov: np.ndarray, inc_t=False):
        raise NotImplementedError
        A = self.A
        dt = self.dt
        timedelta = self.timedelta
        P_t_prev = self.P_t_next
        P_t_next = A @ P_t_prev @ A.T + Q_cov
        self.P_t_next = P_t_next
        self.P_t_prev = P_t_prev
        if inc_t:
            self.t += dt
            self.datetime += self.timedelta

    def propagate_uncertainty_iter(self, Q_cov: np.ndarray, inc_t=False):
        raise NotImplementedError
        dt = self.dt
        timedelta = self.timedelta
        startnodes = self.startnodes
        endnodes = self.endnodes
        indegree = self.indegree
        sub_startnodes = startnodes[(indegree == 0)]
        alpha = self.alpha
        beta = self.beta
        chi = self.chi
        gamma = self.gamma
        P_t_prev = self.P_t_next
        out = np.empty(P_t_prev.shape)
        P_t_next = numba_matmat_par(Q_cov, out, sub_startnodes, endnodes,
                                alpha, beta, chi, indegree)
        self.P_t_next = P_t_next
        self.P_t_prev = P_t_prev
        if inc_t:
            self.t += dt
            self.datetime += self.timedelta

    def time_update(self):
        raise NotImplementedError
        dt = self.dt
        self.t += dt
        self.datetime += self.timedelta

    def simulate(self, dataframe, **kwargs):
        raise NotImplementedError
        assert isinstance(dataframe.index, pd.core.indexes.datetimes.DatetimeIndex)
        # assert (dataframe.index.tz == datetime.timezone.utc)
        assert np.in1d(self.reach_ids, dataframe.columns).all()
        dataframe = dataframe[self.reach_ids]
        dataframe.index = dataframe.index.tz_convert("UTC")
        self.datetime = dataframe.index[0]
        self.t = self.datetime.timestamp()
        for index in dataframe.index:
            p_t_next = dataframe.loc[index, :].values
            self.step(p_t_next, **kwargs)
            yield self

    def simulate_iter(self, dataframe, **kwargs):
        assert isinstance(dataframe.index, pd.core.indexes.datetimes.DatetimeIndex)
        assert (dataframe.index.tz == datetime.timezone.utc)
        assert np.in1d(self.reach_ids, dataframe.columns).all()
        # Execute pre-simulation callbacks
        for _, callback in self.callbacks.items():
            callback.__on_simulation_start__()
        dataframe = dataframe[self.reach_ids]
        for index in dataframe.index:
            timedelta = index - self.datetime
            p_t_next = dataframe.loc[index, :].values
            self.step_iter(p_t_next, timedelta=timedelta, **kwargs)
            yield self
        # Execute post-simulation callbacks
        for _, callback in self.callbacks.items():
            callback.__on_simulation_end__()

    def save_state(self):
        if self.verbose:
            print('Saving state...')
        self.saved_states["datetime"] = self.datetime
        self.saved_states["i_t_next"] = self.i_t_next.copy()
        self.saved_states["o_t_next"] = self.o_t_next.copy()
        for _, callback in self.callbacks.items():
            callback.__on_save_state__()

    def load_state(self):
        if self.verbose:
            print('Loading state...')
        self.datetime = self.saved_states["datetime"]
        self.i_t_next = self.saved_states["i_t_next"]
        self.o_t_next = self.saved_states["o_t_next"]
        for _, callback in self.callbacks.items():
            callback.__on_load_state__()

    def plot(self, ax, *args, **kwargs):
        paths = self.paths
        for path in paths:
            for subpath in path:
                ax.plot(subpath[:,0], subpath[:,1], *args, **kwargs)
    
    def bind_callback(self, callback, key='callback'):
        assert isinstance(callback, BaseCallback)
        self.callbacks[key] = callback

    def unbind_callback(self, key):
        return self.callbacks.pop(key)

    def split(self, indices, inputs, create_state_space=True):
        self = copy.deepcopy(self)
        startnode_indices = np.asarray([np.flatnonzero(self.startnodes == i).item()
                                        for i in indices])
        endnode_indices = self.endnodes[startnode_indices]
        # Cut watershed at indices
        self.endnodes[startnode_indices] = indices
        # Find connected components
        adj = lil_matrix((self.n, self.n), dtype=int)
        for i, j in zip(self.startnodes, self.endnodes):
            adj[j, i] = 1
        n_components, labels = csgraph.connected_components(adj)
        index_map = pd.Series(np.arange(self.n), index=self.startnodes)
        outer_startnodes = labels[startnode_indices].astype(int).tolist()
        outer_endnodes = labels[endnode_indices].astype(int).tolist()
        # Re-order
        new_outer_startnodes = []
        new_outer_endnodes = []
        new_startnode_indices = []
        new_endnode_indices = []
        for k in range(n_components):
            new_outer_startnodes.append(k)
            if k in outer_startnodes:
                pos = outer_startnodes.index(k)
                new_outer_endnodes.append(outer_endnodes[pos])
                new_startnode_indices.append(startnode_indices[pos])
                new_endnode_indices.append(endnode_indices[pos])
            else:
                new_outer_endnodes.append(k)
                new_startnode_indices.append(-1)
                new_endnode_indices.append(-1)
        outer_startnodes = np.asarray(new_outer_startnodes, dtype=int)
        outer_endnodes = np.asarray(new_outer_endnodes, dtype=int)
        startnode_indices = np.asarray(new_startnode_indices, dtype=int)
        endnode_indices = np.asarray(new_endnode_indices, dtype=int)
        outer_indegree = np.bincount(outer_endnodes, minlength=outer_endnodes.size)
        # Create sub-watershed models
        components = {}
        for component in range(n_components):
            components[component] = {}
            selection = (labels == component)
            sub_startnodes = self.startnodes[selection]
            sub_endnodes = self.endnodes[selection]
            new_startnodes = np.arange(len(sub_startnodes))
            node_map = pd.Series(new_startnodes, index=sub_startnodes)
            new_endnodes = node_map.loc[sub_endnodes].values
            sub_index_map = index_map[sub_startnodes].values
            # Create sub-watershed model
            sub_model = copy.deepcopy(self)
            sub_model.startnodes = new_startnodes
            sub_model.endnodes = new_endnodes
            sub_model.n = len(sub_model.startnodes)
            sub_model.indegree = self.indegree[sub_index_map]
            sub_model.reach_ids = np.asarray(self.reach_ids)[sub_index_map].tolist()
            sub_model.dx = self.dx[sub_index_map]
            sub_model.K = self.K[sub_index_map]
            sub_model.X = self.X[sub_index_map]
            sub_model.alpha = self.alpha[sub_index_map]
            sub_model.beta = self.beta[sub_index_map]
            sub_model.chi = self.chi[sub_index_map]
            sub_model.gamma = self.gamma[sub_index_map]
            sub_model.A = np.zeros((sub_model.n, sub_model.n))
            sub_model.B = np.zeros((sub_model.n, sub_model.n))
            sub_model.o_t_prev = self.o_t_prev[sub_index_map]
            sub_model.i_t_prev = self.i_t_prev[sub_index_map]
            sub_model.o_t_next = self.o_t_next[sub_index_map]
            sub_model.i_t_next = self.i_t_next[sub_index_map]
            sub_model.paths = [self.paths[i] for i in sub_index_map]
            if create_state_space:
                sub_model.create_state_space()
            sub_model.save_state()
            components[component]['model'] = sub_model
            components[component]['node_map'] = node_map
            components[component]['input'] = inputs[sub_model.reach_ids].copy()
        # Create connections between sub-watersheds
        for component in range(n_components):
            startnode = outer_startnodes[component]
            endnode = outer_endnodes[component]
            upstream_node_map = components[startnode]['node_map']
            downstream_node_map = components[endnode]['node_map']
            startnode_index = startnode_indices[component]
            endnode_index = endnode_indices[component]
            components[startnode]['terminal_node'] = startnode_index
            components[startnode]['entry_node'] = endnode_index
            if (startnode_index >= 0) and (endnode_index >= 0):
                upstream_model = components[startnode]['model']
                downstream_model = components[endnode]['model']
                exit_index = upstream_node_map[startnode_index]
                entry_index = downstream_node_map[endnode_index]
                downstream_model.indegree[entry_index] -= 1
                downstream_model.i_t_next[entry_index] -= upstream_model.o_t_next[exit_index]
        model_collection = ModelCollection(components, outer_startnodes,
                                           outer_endnodes, startnode_indices,
                                           endnode_indices)
        return model_collection

def _solve_normal_depth(h, Q, B, z, mann_n, So):
    Q_computed = _Q(h, Q, B, z, mann_n, So)
    return Q_computed - Q


def _Q(h, Q, B, z, mann_n, So):
    A = h * (B + z * h)
    P = B + 2 * h * np.sqrt(1 + z**2)
    Q = (np.sqrt(So) / mann_n) * A ** (5 / 3) / P ** (2 / 3)
    return Q

def _dQ_dh(h, Q, B, z, mann_n, So):
    num_0 = 5 * np.sqrt(So) * (h * (B + h * z)) ** (2 / 3) * (B + 2 * h * z)
    den_0 = 3 * mann_n * (B + 2 * h * np.sqrt(z + 1)) ** (2 / 3)
    num_1 = 4 * np.sqrt(So) * np.sqrt(z + 1) * (h * (B + h * z)) ** (5 / 3)
    den_1 = 3 * mann_n * (B + 2 * h * np.sqrt(z + 1)) ** (5 / 3)
    t0 = num_0 / den_0
    t1 = num_1 / den_1
    return t0 - t1


class BaseCallback():
    def __init__(self, *args, **kwargs):
        pass

    def __on_step_start__(self):
        return None
    
    def __on_step_end__(self):
        return None

    def __on_save_state__(self):
        return None

    def __on_load_state__(self):
        return None

    def __on_simulation_start__(self):
        return None

    def __on_simulation_end__(self):
        return None


class ModelCollection():
    def __init__(self, model_dict, outer_startnodes, outer_endnodes, startnode_indices, endnode_indices):
        self.components = model_dict
        self.outer_startnodes = outer_startnodes
        self.outer_endnodes = outer_endnodes
        self.startnode_indices = startnode_indices
        self.endnode_indices = endnode_indices
        self.outer_indegree = np.bincount(outer_endnodes, minlength=outer_endnodes.size)
        self._indegree = self.outer_indegree.copy()
        self.models = {index : model_dict[index]['model'] for index in model_dict}
        self.terminal_nodes = {index : model_dict[index]['terminal_node'] for index in model_dict}
        self.entry_nodes = {index : model_dict[index]['entry_node'] for index in model_dict}
        self.node_map = {index : model_dict[index]['node_map'] for index in model_dict}
        self.inputs = {index : model_dict[index]['input'] for index in model_dict}

    def load_states(self):
        for key in self.models:
            self.models[key].load_state()

    def save_states(self):
        for key in self.models:
            self.models[key].save_state()


class Simulation():
    def __init__(self, model_collection):
        self.model_collection = model_collection
        self.outer_startnodes = model_collection.outer_startnodes
        self.outer_endnodes = model_collection.outer_endnodes
        self.outer_indegree = model_collection.outer_indegree
        self._indegree = model_collection._indegree
        self.models = model_collection.models
        self.terminal_nodes = model_collection.terminal_nodes
        self.entry_nodes = model_collection.entry_nodes
        self.node_map = model_collection.node_map
        self.inputs = model_collection.inputs
        self.outputs = {}

    def simulate(self):
        outer_startnodes = self.outer_startnodes
        outer_endnodes = self.outer_endnodes
        outer_indegree = self.outer_indegree
        multi_outputs = {}
        m = outer_startnodes.size
        outer_indegree_t = outer_indegree.copy()
        outer_indegree_t[outer_startnodes == outer_endnodes] -= 1

        for k in range(m):
            startnode = outer_startnodes[k]
            endnode = outer_endnodes[k]
            while (outer_indegree_t[startnode] == 0):
                model_start = self.models[startnode]
                p_t_start = self.inputs[startnode]
                outputs = {}
                outputs[model_start.datetime] = model_start.o_t_next
                for state in model_start.simulate_iter(p_t_start, inc_t=True):
                    o_t_next = state.o_t_next
                    outputs[state.datetime] = o_t_next
                outputs = pd.DataFrame.from_dict(outputs, orient='index')
                outputs.index = pd.to_datetime(outputs.index, utc=True)
                outputs.columns = p_t_start.columns
                multi_outputs[startnode] = outputs
                print(startnode)
                if startnode != endnode:
                    model_end = self.models[endnode]
                    terminal_node = self.terminal_nodes[startnode]
                    entry_node = self.entry_nodes[startnode]
                    index_out = self.node_map[startnode][terminal_node]
                    index_in = self.node_map[endnode][entry_node]
                    reach_id_out = model_start.reach_ids[index_out]
                    reach_id_in = model_end.reach_ids[index_in]
                    i_t_prev = outputs[reach_id_out].shift(1).iloc[1:].fillna(0.)
                    i_t_next = outputs[reach_id_out].iloc[1:].fillna(0.)
                    gamma_in = model_end.gamma[index_in]
                    alpha_in = model_end.alpha[index_in]
                    beta_in = model_end.beta[index_in]
                    self.inputs[endnode].loc[:, reach_id_in] += (alpha_in
                                                                 * i_t_next.values
                                                                 / gamma_in
                                                               + beta_in
                                                                 * i_t_prev.values
                                                                 / gamma_in)
                    outer_indegree_t[endnode] -= 1
                    startnode = endnode
                    endnode = outer_endnodes[startnode]
                else:
                    break
        return multi_outputs

class AsyncSimulation(Simulation):
    def __init__(self, model_collection):
        return super().__init__(model_collection)
    
    async def simulate(self, verbose=False):
        indegree = self.outer_indegree.copy()
        indegree[self.outer_endnodes == self.outer_startnodes] -= 1
        self._indegree = indegree
        m = len(self.models)
        try:
            asyncio.get_running_loop()
            loop_running = True
        except RuntimeError:
            loop_running = False
        if loop_running:
            await self._main(verbose=verbose)
        else:
            asyncio.run(self._main(verbose=verbose))
        return self.outputs

    async def _main(self, verbose=False):
        indegree = self._indegree
        async with asyncio.TaskGroup() as taskgroup:
            for index, predecessors in enumerate(indegree):
                if predecessors == 0:
                    model = self.models[index]
                    inputs = self.inputs[index]
                    taskgroup.create_task(self._simulate(taskgroup, model,
                                                         inputs, index, 
                                                         verbose))

    async def _simulate(self, taskgroup, model, inputs, index, verbose=False):
        if verbose:
            print(f'Started job for sub-watershed {index}')
        start_time = model.datetime
        outputs = {}
        outputs[start_time] = model.o_t_next
        for state in model.simulate_iter(inputs, inc_t=True):
            current_time = state.datetime
            o_t_next = state.o_t_next
            outputs[current_time] = o_t_next
        outputs = pd.DataFrame.from_dict(outputs, orient='index')
        outputs.index = pd.to_datetime(outputs.index, utc=True)
        outputs.columns = inputs.columns
        self.outputs[index] = outputs
        taskgroup.create_task(self._accumulate(taskgroup, outputs, index, verbose))

    async def _accumulate(self, taskgroup, outputs, index, verbose=False):
        indegree = self._indegree
        startnode = index
        endnode = self.outer_endnodes[startnode]
        if startnode != endnode:
            model_start = self.models[startnode]
            model_end = self.models[endnode]
            inputs = self.inputs[endnode]
            terminal_node = self.terminal_nodes[startnode]
            entry_node = self.entry_nodes[startnode]
            index_out = self.node_map[startnode][terminal_node]
            index_in = self.node_map[endnode][entry_node]
            reach_id_out = model_start.reach_ids[index_out]
            reach_id_in = model_end.reach_ids[index_in]
            i_t_prev = outputs[reach_id_out].shift(1).iloc[1:].fillna(0.)
            i_t_next = outputs[reach_id_out].iloc[1:].fillna(0.)
            gamma_in = model_end.gamma[index_in]
            alpha_in = model_end.alpha[index_in]
            beta_in = model_end.beta[index_in]
            inputs.loc[:, reach_id_in] += (alpha_in * i_t_next.values / gamma_in
                                        + beta_in * i_t_prev.values / gamma_in)
            indegree[endnode] -= 1
            if (indegree[endnode] == 0):
                taskgroup.create_task(self._simulate(taskgroup, model_end,
                                                     inputs, endnode, verbose))
        if verbose:
            print(f'Finished job for sub-watershed {index}')

class MultiThreadedSimulation(AsyncSimulation):
    def __init__(self, model_collection, max_workers=None):
        super().__init__(model_collection)
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers)

    async def simulate(self, verbose=False):
        indegree = self.outer_indegree.copy()
        indegree[self.outer_endnodes == self.outer_startnodes] -= 1
        self._indegree = indegree
        m = len(self.models)
        try:
            loop = asyncio.get_running_loop()
            self.loop = loop
            loop_running = True
        except RuntimeError:
            loop_running = False
            raise
        await self._main(verbose=verbose)
        self.flush_executor()
        return self.outputs
    
    async def _main(self, verbose=False):
        indegree = self._indegree
        async with asyncio.TaskGroup() as taskgroup:
            for index, predecessors in enumerate(indegree):
                if predecessors == 0:
                    model = self.models[index]
                    inputs = self.inputs[index]
                    taskgroup.create_task(self._simulate(taskgroup, model,
                                                         inputs, index,
                                                         verbose))

    async def _simulate(self, taskgroup, model, inputs, index, loop, verbose=False):
        if verbose:
            print(f'Started job for sub-watershed {index}')
        executor = self.executor
        loop = self.loop
        outputs, model_post = await loop.run_in_executor(executor, _inner_simulation,
                                                         model, inputs)
        self.outputs[index] = outputs
        model.saved_states.update(model_post.saved_states)
        taskgroup.create_task(self._accumulate(taskgroup, outputs, index, verbose))

    def flush_executor(self):
        self.executor.shutdown()
        self.executor = ThreadPoolExecutor(self.max_workers)

class MultiProcessSimulation(MultiThreadedSimulation):
    def __init__(self, model_collection, max_workers=None):
        super().__init__(model_collection)
        self.max_workers = max_workers
        self.executor = ProcessPoolExecutor(max_workers)

    def flush_executor(self):
        self.executor.shutdown()
        self.executor = ProcessPoolExecutor(self.max_workers)


def _inner_simulation(model, inputs):
    start_time = model.datetime
    outputs = {}
    outputs[start_time] = model.o_t_next
    for state in model.simulate_iter(inputs, inc_t=True):
        current_time = state.datetime
        o_t_next = state.o_t_next
        outputs[current_time] = o_t_next
    outputs = pd.DataFrame.from_dict(outputs, orient='index')
    outputs.index = pd.to_datetime(outputs.index, utc=True)
    outputs.columns = inputs.columns
    return outputs, model


class CheckPoint(BaseCallback):
    def __init__(self, model, checkpoint_time=None, timedelta=None):
        self.model = model
        if checkpoint_time is None:
            if timedelta is None:
                raise ValueError('Either `checkpoint_time` or `timedelta` must not be `None`.')
            else:
                checkpoint_time = model.datetime + datetime.timedelta(seconds=timedelta)
        self.checkpoint_time = checkpoint_time
        self.timedelta = timedelta
        self.model_saved = False

    def __on_simulation_start__(self):
        timedelta = self.timedelta
        if timedelta is None:
            return None
        else:
            checkpoint_time = self.model.datetime + datetime.timedelta(seconds=timedelta)
            self.set_checkpoint(checkpoint_time)
    
    def __on_step_end__(self):
        checkpoint_time = self.checkpoint_time
        current_time = self.model.datetime
        if (current_time >= checkpoint_time) and (not self.model_saved):
            self.model.save_state()
            self.model_saved = True

    def __on_save_state__(self):
        model = self.model
        for _, callback in model.callbacks.items():
            if hasattr(callback, 'save_state'):
                callback.save_state()

    def __on_load_state__(self):
        model = self.model
        for _, callback in model.callbacks.items():
            if hasattr(callback, 'load_state'):
                callback.load_state()
    
    def set_checkpoint(self, checkpoint_time):
        self.checkpoint_time = checkpoint_time
        self.model_saved = False


class KalmanFilter(BaseCallback):
    def __init__(self, model, measurements, Q_cov, R_cov, P_t_init):
        self.model = model
        self.measurements = measurements
        self.Q_cov = Q_cov
        self.R_cov = R_cov
        self.P_t_next = P_t_init
        self.reach_ids = model.reach_ids
        self.gage_reach_ids = measurements.columns
        self.datetime = copy.deepcopy(model.datetime)
        self.num_measurements = measurements.shape[1]

        assert isinstance(measurements.index, pd.core.indexes.datetimes.DatetimeIndex)
        assert (measurements.index.tz == datetime.timezone.utc)
        assert np.in1d(measurements.columns, self.reach_ids).all()

        reach_index_map = pd.Series(np.arange(len(self.reach_ids)), index=self.reach_ids)
        reach_indices = reach_index_map.loc[self.gage_reach_ids].values.astype(int)
        s = np.zeros(model.n, dtype=bool)
        s[reach_indices] = True
        self.s = s
        # NOTE: Sorting of input data done here
        permutations = np.argsort(reach_indices)
        is_sorted = (permutations == np.arange(self.num_measurements)).all()
        if not is_sorted:
            logger.warning('Measurement indices not sorted. Permuting columns...')
        reach_indices = reach_indices[permutations]
        self.reach_indices = reach_indices
        self.measurements = self.measurements.iloc[:, permutations]
        self.R_cov = self.R_cov[permutations, :][:, permutations]

        self.saved_states = {}
        self.save_state()

    def __on_simulation_start__(self):
        return None
        # TODO: Double-check for off-by-one error
        if self.model.datetime > self.latest_timestamp:
            return None
        else:
            return self.filter()

    def __on_step_end__(self):
        # TODO: Double-check for off-by-one error
        if self.model.datetime > self.latest_timestamp:
            return None
        else:
            return self.filter()

    @property
    def latest_measurement(self):
        return self.measurements.iloc[-1, :].values

    @property
    def latest_timestamp(self):
        return self.measurements.index[-1]

    def interpolate_input(self, datetime, method='linear'):
        datetime = float(datetime.value)
        datetimes = self.measurements.index.astype(int).astype(float)
        samples = self.measurements.values
        if method == 'linear':
            method_code = 1
        elif method == 'nearest':
            method_code = 0
        else:
            raise ValueError
        return interpolate_sample(datetime, datetimes, samples, method=method_code)

    def save_state(self):
        self.saved_states["datetime"] = self.datetime
        self.saved_states["P_t_next"] = self.P_t_next.copy()

    def load_state(self):
        self.datetime = self.saved_states["datetime"]
        self.P_t_next = self.saved_states["P_t_next"]

    def filter(self):
        # Implements KF after step is called
        P_t_prev = self.P_t_next
        i_t_next = self.model.i_t_next
        o_t_next = self.model.o_t_next
        Q_cov = self.Q_cov
        R_cov = self.R_cov
        measurements = self.measurements
        s = self.s
        startnodes = self.model.startnodes
        endnodes = self.model.endnodes
        indegree = self.model.indegree
        sub_startnodes = startnodes[(indegree == 0)]
        alpha = self.model.alpha
        beta = self.model.beta
        chi = self.model.chi
        gamma = self.model.gamma
        datetime = self.model.datetime
        # Computed parameters
        Z = measurements.loc[datetime].values
        dz = Z - o_t_next[s]
        # Compute prior covariance
        out = np.empty(P_t_prev.shape)
        P_t_next = numba_matmat_par(P_t_prev, out, sub_startnodes, endnodes,
                                    alpha, beta, chi, indegree)
        P_t_next += Q_cov
        # Compute gain and posterior covariance
        K = P_t_next[:,s] @ np.linalg.inv(P_t_next[s][:,s] + R_cov)
        gain = K @ dz
        P_t_next = P_t_next - K @ P_t_next[s]
        # Apply gain
        i_t_gain, o_t_gain = _apply_gain(sub_startnodes, endnodes, gain, indegree)
        i_t_next += i_t_gain
        o_t_next += o_t_gain
        # Save posterior estimates
        self.model.i_t_next = i_t_next
        self.model.o_t_next = o_t_next        
        self.P_t_next = P_t_next
        # Update time
        self.datetime = datetime


class KalmanSmoother(KalmanFilter):
    def __init__(self, model, measurements, Q_cov, R_cov):
        super().__init__(model, measurements, Q_cov, R_cov)
        N = len(measurements) - 1
        self.N = N
        self.datetimes = [model.datetime]
        self.P_f = {}
        self.P_p = {}
        self.i_hat_f = {}
        self.o_hat_f = {}
        self.i_hat_f[model.datetime] = self.model.i_t_next.copy()
        self.o_hat_f[model.datetime] = self.model.o_t_next.copy()
        self.i_hat_p = {}
        self.o_hat_p = {}
        self.i_hat_p[model.datetime] = self.model.i_t_next.copy()
        self.o_hat_p[model.datetime] = self.model.o_t_next.copy()
        self.P_f[model.datetime] = self.model.P_t_next.copy()
        self.P_p[model.datetime] = self.model.P_t_next.copy()
        self.i_hat_s = None
        self.o_hat_s = None

    def __on_step_end__(self):
        # TODO: Double-check for off-by-one error
        if self.model.datetime > self.latest_timestamp:
            return None
        else:
            return self.filter()

    def __on_simulation_end__(self):
        return self.smooth()
    
    def filter(self):
        # Implements KF after step is called
        P_t_prev = self.model.P_t_next
        i_t_prior = self.model.i_t_next
        o_t_prior = self.model.o_t_next
        Q_cov = self.Q_cov
        R_cov = self.R_cov
        measurements = self.measurements
        s = self.s
        startnodes = self.model.startnodes
        endnodes = self.model.endnodes
        indegree = self.model.indegree
        sub_startnodes = startnodes[(indegree == 0)]
        alpha = self.model.alpha
        beta = self.model.beta
        chi = self.model.chi
        gamma = self.model.gamma
        datetime = self.model.datetime
        # Computed parameters
        Z = measurements.loc[datetime].values
        dz = Z - o_t_prior[s]
        # Compute prior covariance
        out = np.empty(P_t_prev.shape)
        P_t_prior = numba_matmat_par(P_t_prev, out, sub_startnodes, endnodes,
                                    alpha, beta, chi, indegree)
        P_t_prior += Q_cov
        # Compute gain and posterior covariance
        K = P_t_prior[:,s] @ np.linalg.inv(P_t_prior[s][:,s] + R_cov)
        gain = K @ dz
        P_t_next = P_t_prior - K @ P_t_prior[s]
        # Apply gain
        i_t_gain, o_t_gain = _apply_gain(sub_startnodes, endnodes, gain, indegree)
        i_t_next = i_t_prior + i_t_gain
        o_t_next = o_t_prior + o_t_gain
        # Save posterior estimates
        self.model.i_t_next = i_t_next
        self.model.o_t_next = o_t_next        
        self.model.P_t_prev = P_t_prev
        self.model.P_t_next = P_t_next
        # Save outputs
        self.i_hat_f[datetime] = i_t_next
        self.i_hat_p[datetime] = i_t_prior
        self.o_hat_f[datetime] = o_t_next
        self.o_hat_p[datetime] = o_t_prior
        self.P_p[datetime] = P_t_prior
        self.P_f[datetime] = P_t_next
        self.datetimes.append(datetime)
        # Update time
        self.datetime = datetime

    def smooth(self):
        P_p = self.P_p
        P_f = self.P_f
        datetimes = self.datetimes
        N = len(datetimes) - 1
        i_hat_f = self.i_hat_f
        o_hat_f = self.o_hat_f
        i_hat_p = self.i_hat_p
        o_hat_p = self.o_hat_p
        P_s = {}
        i_hat_s = {}
        o_hat_s = {}
        P_s[self.datetime] = P_f[self.datetime]
        i_hat_s[self.datetime] = i_hat_f[self.datetime]
        o_hat_s[self.datetime] = o_hat_f[self.datetime]
        startnodes = self.model.startnodes
        endnodes = self.model.endnodes
        indegree = self.model.indegree
        sub_startnodes = startnodes[(indegree == 0)]
        alpha = self.model.alpha
        beta = self.model.beta
        chi = self.model.chi
        gamma = self.model.gamma
        # Make this more efficient later
        A = self.model.A
        for k in reversed(range(N)):
            t = datetimes[k]
            tp1 = datetimes[k+1]
            P_f_t = P_f[t]
            out = np.empty(P_f_t.shape)
            A_Pf = _ap_par(P_f_t, out, sub_startnodes, endnodes, alpha, beta, chi, indegree)
            J = np.linalg.solve(P_p[tp1], A_Pf).T
            i_t_s = i_hat_f[t] + J @ (i_hat_s[tp1] - i_hat_p[tp1])
            o_t_s = o_hat_f[t] + J @ (o_hat_s[tp1] - o_hat_p[tp1])
            P = P_f[t] + J @ (P_s[tp1] - P_p[tp1])
            i_hat_s[t] = i_t_s
            o_hat_s[t] = o_t_s
            P_s[t] = P
        i_hat_s = pd.DataFrame.from_dict(i_hat_s, orient='index')
        i_hat_s.columns = self.model.reach_ids
        o_hat_s = pd.DataFrame.from_dict(o_hat_s, orient='index')
        o_hat_s.columns = self.model.reach_ids
        self.i_hat_s = i_hat_s
        self.o_hat_s = o_hat_s
        self.P_s = P_s

class ExpectationMaximization(KalmanSmoother):
    def __init__(self, model, measurements, Q_cov, R_cov, regularization=0.):
        super().__init__(model, measurements, Q_cov, R_cov)
        self.regularization = regularization
        self.K = model.K.copy()
        self.X = model.X.copy()
        self.G = None
        self.a = None
        self.C = None
        self.b = None
        self.result = None
    
    def __on_simulation_end__(self):
        self.smooth()
        return self.solve_parameters()

    def solve_parameters(self):
        model = self.model
        datetimes = self.datetimes
        i_hat = self.i_hat_s.values
        o_hat = self.o_hat_s.values
        p_t = (self.measurements.reindex(model.reach_ids, 
               axis=1).fillna(0.).loc[datetimes, :].values)
        dt = model.dt
        regularization = self.regularization
        # Minimization terms
        m = 1 + 2 * model.n
        T1 = o_hat[1:] - o_hat[:-1]
        T2 = i_hat[:-1] + o_hat[:-1] - i_hat[1:] - o_hat[1:]
        T3 = (0.5 * o_hat[1:] + 0.5 * o_hat[:-1] 
            - 0.5 * i_hat[1:] - 0.5 * i_hat[:-1] - p_t[:-1])
        # Inner blocks
        ul = (T1**2).sum(axis=0)
        ur = (T1 * T2).sum(axis=0)
        ll = ur
        lr = (T2**2).sum(axis=0)
        diag = np.diag(np.column_stack([ul, lr]).ravel())
        offdiag = np.diag(np.column_stack([ur, np.zeros(ur.size)]).ravel()[:-1], k=1)
        # Outer border
        uul = (T3**2).sum()
        lll = (T3 * T1).sum(axis=0)
        llr = (T3 * T2).sum(axis=0)
        c = np.column_stack([lll, llr]).ravel()
        # Set up quadratic program
        # G matrix
        G = np.zeros((m, m))
        G[0, 0] = uul
        G[1:, 0] = c
        G[0, 1:] = c
        G[1:, 1:] = diag + offdiag + offdiag.T
        # C matrix
        C_block = np.array([0.5, -1, 0, 1, 1, 0, 1, -1]).reshape(4, 2)
        C = np.zeros((1 + 4*model.n, 1 + 2*model.n))
        C[0,0] = 1.
        C[1:, 1:] = scipy.linalg.block_diag(*[C_block] * model.n)
        # b vector
        b = 0. * np.ones(1 + 4 * model.n)
        b[0] = dt
        b[4::4] = dt / 2
        # a vector
        a = np.zeros(m)
        G_reg = G + np.eye(m) * regularization
        result = quadprog.solve_qp(G_reg, a, C.T, b, meq=1)
        x = result[0]
        theta = x[1:].reshape(-1, 2)
        K = theta[:,0]
        X = theta[:,1] / theta[:,0]
        self.K = K
        self.X = X
        self.G = G
        self.a = a
        self.C = C
        self.b = b
        self.result = result


class KalmanSmootherIO(KalmanSmoother):
    def __init__(self, model, measurements, Q_cov, R_cov, temp_file):
        super(KalmanSmoother, self).__init__(model, measurements, Q_cov, R_cov)
        epoch = str(model.datetime.value)
        N = len(measurements) - 1
        self.N = N
        self.datetimes = [model.datetime]
        self.i_hat_f = {}
        self.o_hat_f = {}
        self.i_hat_p = {}
        self.o_hat_p = {}
        self.i_hat_f[model.datetime] = self.model.i_t_next.copy()
        self.o_hat_f[model.datetime] = self.model.o_t_next.copy()
        self.i_hat_p[model.datetime] = self.model.i_t_next.copy()
        self.o_hat_p[model.datetime] = self.model.o_t_next.copy()
        self.temp_file = temp_file
        P_f = pd.DataFrame(self.model.P_t_next)
        P_p = pd.DataFrame(self.model.P_t_next)
        P_f.to_hdf(f'{temp_file}', key=f'Pf__{epoch}', mode='a')
        P_p.to_hdf(f'{temp_file}', key=f'Pp__{epoch}', mode='a')
        self.i_hat_s = None
        self.o_hat_s = None

    def __on_simulation_end__(self):
        self.smooth()
        os.remove(self.temp_file)

    def filter(self):
        # Implements KF after step is called
        P_t_prev = self.model.P_t_next
        i_t_prior = self.model.i_t_next
        o_t_prior = self.model.o_t_next
        Q_cov = self.Q_cov
        R_cov = self.R_cov
        measurements = self.measurements
        s = self.s
        startnodes = self.model.startnodes
        endnodes = self.model.endnodes
        indegree = self.model.indegree
        sub_startnodes = startnodes[(indegree == 0)]
        alpha = self.model.alpha
        beta = self.model.beta
        chi = self.model.chi
        gamma = self.model.gamma
        datetime = self.model.datetime
        epoch = str(self.model.datetime.value)
        # Computed parameters
        Z = measurements.loc[datetime].values
        dz = Z - o_t_prior[s]
        # Compute prior covariance
        out = np.empty(P_t_prev.shape)
        P_t_prior = numba_matmat_par(P_t_prev, out, sub_startnodes, endnodes,
                                    alpha, beta, chi, indegree)
        P_t_prior += Q_cov
        # Compute gain and posterior covariance
        K = P_t_prior[:,s] @ np.linalg.inv(P_t_prior[s][:,s] + R_cov)
        gain = K @ dz
        P_t_next = P_t_prior - K @ P_t_prior[s]
        # Apply gain
        i_t_gain, o_t_gain = _apply_gain(sub_startnodes, endnodes, gain, indegree)
        i_t_next = i_t_prior + i_t_gain
        o_t_next = o_t_prior + o_t_gain
        # Save posterior estimates
        self.model.i_t_next = i_t_next
        self.model.o_t_next = o_t_next        
        self.model.P_t_prev = P_t_prev
        self.model.P_t_next = P_t_next
        # Save outputs
        self.i_hat_f[datetime] = i_t_next
        self.i_hat_p[datetime] = i_t_prior
        self.o_hat_f[datetime] = o_t_next
        self.o_hat_p[datetime] = o_t_prior
        pd.DataFrame(P_t_prior).to_hdf(f'{self.temp_file}', key=f'Pp__{epoch}', mode='a')
        pd.DataFrame(P_t_next).to_hdf(f'{self.temp_file}', key=f'Pf__{epoch}', mode='a')
        self.datetimes.append(datetime)
        # Update time
        self.datetime = datetime

    def smooth(self):
        temp_file = self.temp_file
        datetimes = self.datetimes
        N = len(datetimes) - 1
        i_hat_f = self.i_hat_f
        o_hat_f = self.o_hat_f
        i_hat_p = self.i_hat_p
        o_hat_p = self.o_hat_p
        i_hat_s = {}
        o_hat_s = {}
        P_s_tp1 = pd.read_hdf(f'{temp_file}', key=f'Pf__{self.datetime.value}').values
        i_hat_s[self.datetime] = i_hat_f[self.datetime]
        o_hat_s[self.datetime] = o_hat_f[self.datetime]
        startnodes = self.model.startnodes
        endnodes = self.model.endnodes
        indegree = self.model.indegree
        sub_startnodes = startnodes[(indegree == 0)]
        alpha = self.model.alpha
        beta = self.model.beta
        chi = self.model.chi
        for k in reversed(range(N)):
            t = datetimes[k]
            tp1 = datetimes[k+1]
            P_f_t = pd.read_hdf(f'{temp_file}', key=f'Pf__{t.value}').values
            P_p_tp1 = pd.read_hdf(f'{temp_file}', key=f'Pp__{tp1.value}').values
            out = np.empty(P_f_t.shape)
            A_Pf = _ap_par(P_f_t, out, sub_startnodes, endnodes, alpha, beta, chi, indegree)
            J = np.linalg.solve(P_p_tp1, A_Pf).T
            i_t_s = i_hat_f[t] + J @ (i_hat_s[tp1] - i_hat_p[tp1])
            o_t_s = o_hat_f[t] + J @ (o_hat_s[tp1] - o_hat_p[tp1])
            P = P_f_t + J @ (P_s_tp1 - P_p_tp1)
            i_hat_s[t] = i_t_s
            o_hat_s[t] = o_t_s
            P_s_tp1 = P
        i_hat_s = pd.DataFrame.from_dict(i_hat_s, orient='index')
        i_hat_s.columns = self.model.reach_ids
        o_hat_s = pd.DataFrame.from_dict(o_hat_s, orient='index')
        o_hat_s.columns = self.model.reach_ids
        self.i_hat_s = i_hat_s
        self.o_hat_s = o_hat_s


class ExpectationMaximizationIO(KalmanSmootherIO):
    def __init__(self, model, measurements, Q_cov, R_cov, temp_file, regularization=0.):
        super().__init__(model, measurements, Q_cov, R_cov, temp_file)
        self.regularization = regularization
        self.K = model.K.copy()
        self.X = model.X.copy()
        self.G = None
        self.a = None
        self.C = None
        self.b = None
        self.result = None

    def __on_simulation_end__(self):
        self.smooth()
        os.remove(self.temp_file)
        return self.solve_parameters()

    def solve_parameters(self):
        model = self.model
        datetimes = self.datetimes
        i_hat = self.i_hat_s.values
        o_hat = self.o_hat_s.values
        p_t = (self.measurements.reindex(model.reach_ids, 
               axis=1).fillna(0.).loc[datetimes, :].values)
        dt = model.dt
        regularization = self.regularization
        # Minimization terms
        m = 1 + 2 * model.n
        T1 = o_hat[1:] - o_hat[:-1]
        T2 = i_hat[:-1] + o_hat[:-1] - i_hat[1:] - o_hat[1:]
        T3 = (0.5 * o_hat[1:] + 0.5 * o_hat[:-1] 
            - 0.5 * i_hat[1:] - 0.5 * i_hat[:-1] - p_t[:-1])
        # Inner blocks
        ul = (T1**2).sum(axis=0)
        ur = (T1 * T2).sum(axis=0)
        ll = ur
        lr = (T2**2).sum(axis=0)
        diag = np.diag(np.column_stack([ul, lr]).ravel())
        offdiag = np.diag(np.column_stack([ur, np.zeros(ur.size)]).ravel()[:-1], k=1)
        # Outer border
        uul = (T3**2).sum()
        lll = (T3 * T1).sum(axis=0)
        llr = (T3 * T2).sum(axis=0)
        c = np.column_stack([lll, llr]).ravel()
        # Set up quadratic program
        # G matrix
        G = np.zeros((m, m))
        G[0, 0] = uul
        G[1:, 0] = c
        G[0, 1:] = c
        G[1:, 1:] = diag + offdiag + offdiag.T
        # C matrix
        C_block = np.array([0.5, -1, 0, 1, 1, 0, 1, -1]).reshape(4, 2)
        C = np.zeros((1 + 4*model.n, 1 + 2*model.n))
        C[0,0] = 1.
        C[1:, 1:] = scipy.linalg.block_diag(*[C_block] * model.n)
        # b vector
        b = 0. * np.ones(1 + 4 * model.n)
        b[0] = dt
        b[4::4] = dt / 2
        # a vector
        a = np.zeros(m)
        G_reg = G + np.eye(m) * regularization
        result = quadprog.solve_qp(G_reg, a, C.T, b, meq=1)
        x = result[0]
        theta = x[1:].reshape(-1, 2)
        K = theta[:,0]
        X = theta[:,1] / theta[:,0]
        self.K = K
        self.X = X
        self.G = G
        self.a = a
        self.C = C
        self.b = b
        self.result = result


@njit # Uncomment for large networks
def _ax_bu(startnodes, endnodes, alpha, beta, chi, gamma, i_t_prev, o_t_prev, q_t_next, indegree):
    n = endnodes.size
    m = startnodes.size
    i_t_next = np.zeros(n, dtype=np.float64)
    o_t_next = np.zeros(n, dtype=np.float64)
    indegree_t = indegree.copy()
    # Simulate output
    for k in range(m):
        startnode = startnodes[k]
        endnode = endnodes[startnode]
        while(indegree_t[startnode] == 0):
            alpha_i = alpha[startnode]
            beta_i = beta[startnode]
            chi_i = chi[startnode]
            gamma_i = gamma[startnode]
            o_t_next[startnode] += (alpha_i * i_t_next[startnode]
                                    + beta_i * i_t_prev[startnode]
                                    + chi_i * o_t_prev[startnode]
                                    + gamma_i * q_t_next[startnode])
            if startnode != endnode:
                i_t_next[endnode] += o_t_next[startnode]
            indegree_t[endnode] -= 1
            startnode = endnode
            endnode = endnodes[startnode]
    return i_t_next, o_t_next

@njit # Uncomment for large networks
def _ax(startnodes, endnodes, alpha, beta, chi, i_t_prev, o_t_prev, indegree):
    n = endnodes.size
    m = startnodes.size
    i_t_next = np.zeros(n, dtype=np.float64)
    o_t_next = np.zeros(n, dtype=np.float64)
    indegree_t = indegree.copy()
    # Simulate output
    for k in range(m):
        startnode = startnodes[k]
        endnode = endnodes[startnode]
        while(indegree_t[startnode] == 0):
            alpha_i = alpha[startnode]
            beta_i = beta[startnode]
            chi_i = chi[startnode]
            o_t_next[startnode] += (alpha_i * i_t_next[startnode]
                                    + beta_i * i_t_prev[startnode]
                                    + chi_i * o_t_prev[startnode])
            if startnode != endnode:
                i_t_next[endnode] += o_t_next[startnode]
            indegree_t[endnode] -= 1
            startnode = endnode
            endnode = endnodes[startnode]
    return i_t_next, o_t_next

@njit # Uncomment for large networks
def _apply_gain(startnodes, endnodes, gain, indegree):
    n = endnodes.size
    m = startnodes.size
    i_t_next = np.zeros(n, dtype=np.float64)
    o_t_next = np.zeros(n, dtype=np.float64)
    indegree_t = indegree.copy()
    # Simulate output
    for k in range(m):
        startnode = startnodes[k]
        endnode = endnodes[startnode]
        while(indegree_t[startnode] == 0):
            o_t_next[startnode] += gain[startnode]
            if startnode != endnode:
                i_t_next[endnode] += o_t_next[startnode]
            indegree_t[endnode] -= 1
            startnode = endnode
            endnode = endnodes[startnode]
    return i_t_next, o_t_next

@njit
def numba_init_inflows(a, indices, b):
    n = len(indices)
    for i in range(n):
        k = indices[i]
        a[k] += b[i]

@njit
def _ap(P, out, startnodes, endnodes, alpha, beta, chi,
                 indegree):
    m, n = P.shape
    assert (m == n)
    for i in range(n):
        o_t_prev = P[:, i]
        i_t_prev = np.zeros(n, dtype=np.float64)
        numba_init_inflows(i_t_prev, endnodes, o_t_prev)
        _, o_t_next = _ax(startnodes, endnodes, alpha, beta, chi,
                          i_t_prev, o_t_prev, indegree)
        out[:, i] = o_t_next
    return out

@njit(parallel=True)
def _ap_par(P, out, startnodes, endnodes, alpha, beta, chi,
                 indegree):
    m, n = P.shape
    assert (m == n)
    for i in prange(n):
        o_t_prev = P[:, i]
        i_t_prev = np.zeros(n, dtype=np.float64)
        numba_init_inflows(i_t_prev, endnodes, o_t_prev)
        _, o_t_next = _ax(startnodes, endnodes, alpha, beta, chi,
                          i_t_prev, o_t_prev, indegree)
        out[:, i] = o_t_next
    return out


@njit
def numba_matmat(P, out, startnodes, endnodes, alpha, beta, chi,
                 indegree):
    m, n = P.shape
    assert (m == n)
    for i in range(n):
        o_t_prev = P[:, i]
        i_t_prev = np.zeros(n, dtype=np.float64)
        numba_init_inflows(i_t_prev, endnodes, o_t_prev)
        _, o_t_next = _ax(startnodes, endnodes, alpha, beta, chi,
                          i_t_prev, o_t_prev, indegree)
        out[:, i] = o_t_next
    out = out.T
    for i in range(n):
        o_t_prev = out[:, i]
        i_t_prev = np.zeros(n, dtype=np.float64)
        numba_init_inflows(i_t_prev, endnodes, o_t_prev)
        _, o_t_next = _ax(startnodes, endnodes, alpha, beta, chi,
                          i_t_prev, o_t_prev, indegree)
        out[:, i] = o_t_next
    return out

@njit(parallel=True)
def numba_matmat_par(P, out, startnodes, endnodes, alpha, beta, chi,
                     indegree):
    m, n = P.shape
    assert (m == n)
    for i in prange(n):
        o_t_prev = P[:, i]
        i_t_prev = np.zeros(n, dtype=np.float64)
        numba_init_inflows(i_t_prev, endnodes, o_t_prev)
        _, o_t_next = _ax(startnodes, endnodes, alpha, beta, chi,
                          i_t_prev, o_t_prev, indegree)
        out[:, i] = o_t_next
    out = out.T
    for i in prange(n):
        o_t_prev = out[:, i]
        i_t_prev = np.zeros(n, dtype=np.float64)
        numba_init_inflows(i_t_prev, endnodes, o_t_prev)
        _, o_t_next = _ax(startnodes, endnodes, alpha, beta, chi,
                          i_t_prev, o_t_prev, indegree)
        out[:, i] = o_t_next
    return out

