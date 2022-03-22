import numpy as np
import numba


@numba.jit(numba.float64[:](numba.float64, numba.float64, numba.float64),
           fastmath=True, cache=True, nopython=True, nogil=True)
def init_timeline(time_0, time_delta, time_end=1.):
    """
    Initializes timeline for non-stationary numeric approximation of integral equation in forward time formulation

    :param time_0: real variable -- initial time for numeric approximation
    :param time_delta: real variable -- The time interval between discrete time counts
    :param time_end:    real param -- time where you need to end calculations

    :return: np.1darray -- problem timeline
    """
    timeline = np.arange(time_0, time_end + time_delta, time_delta)
    return timeline


@numba.jit(fastmath=True, cache=True, nopython=True, nogil=True)
def find_timeline_tau_indexes_between(tau, time_delta):
    """
    Finds upper shift of array index and lower shift of array index for time shift by tau from timeline

    :param tau: time shift for time from timeline
    :param time_delta:  timeline discretization
    :return: tuple of indexes, where lower shift and upper shift are placed correspondingly
    """
    upper_index = int(np.floor(tau/time_delta))
    lower_index = int(np.ceil(tau/time_delta))
    indexes = (lower_index, upper_index)
    return indexes


@numba.jit(fastmath=True, cache=True, nopython=True, nogil=True)
def find_time_proportions(time_shifted, time_lower, time_delta):
    """
    find linear proportions

    :param time_shifted:
    :param time_lower:
    :param time_delta:
    :return:
    """
    beta = np.abs(time_shifted - time_lower) / time_delta
    alpha = 1 - beta
    proportions = (alpha, beta)
    return proportions


def _test():
    tau = 0.025
    time_delta = 0.004

    timeline = init_timeline(time_0=0., time_delta=time_delta, time_end=1.)
    time = timeline[53]

    indexes = find_timeline_tau_indexes_between(tau, time_delta)
    print("time =", time,"minus tau =", tau, " t =", time - tau," is between ", (timeline[53 - indexes[0]], timeline[53 - indexes[1]]))


if __name__ == "__main__":
    _test()