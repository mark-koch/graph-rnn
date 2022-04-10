'''Code for calculating maximum mean discrepancy (MMD) metric for graph similarity.'''

from scipy.stats import wasserstein_distance
from pyemd import emd
from math import exp
import numpy as np


def expected_discrepancy(dist1, dist2, kernel_func, sigma=1.0):
    """
    Compute expected value of the discrepancy metric between two samples.

    To get the expected value, compute discrepancy between all subsets of samples
    and take the average of these discrepancy.

    :param dist1: a 1-D array of numbers
    :param dist2: a 1-D array of numbers
    :param kernel_func: the kernel function to use in computing MMD
    :param sigma: the sigma used for computing the kernel function
    """
    total_discrepancy = 0
    count = 0
    for data_pt1 in dist1:
        for data_pt2 in dist2:
            # print("kernel result:", kernel_func(data_pt1, data_pt2, sigma))
            total_discrepancy += kernel_func(data_pt1, data_pt2, sigma)
            count += 1

    # print("total_discrepancy:", total_discrepancy)
    # print("length:", count)
    avg_discrepancy = total_discrepancy / count
    return avg_discrepancy


def gaussian_kernel(dist_p, dist_q, sigma=1.0):
    """Gaussian kernel function."""
    # distance_matrix = 
    # wass_distance = emd(dist_p, dist_q, distance_matrix) 
    # return exp(-wass_distance * wass_distance / (2 * (sigma ** 2)))
    wass_distance = wasserstein_distance(dist_p, dist_q)
    result = exp(wass_distance / (2 * (sigma ** 2)))
    # print("kernel result:", result)
    return result


def mmd(dist1, dist2, kernel_func=gaussian_kernel, sigma=1.0):
    """
    Compute squared maximum mean discrepancy (MMD) metric between two distributions.

    :param dist1: a 1-D array of numbers
    :param dist2: a 1-D array of numbers
    :param kernel_func: the kernel function to use in computing MMD
    :param sigma: the sigma used for computing the kernel function
    """
    ex_x_y_from_p_of_k = expected_discrepancy(dist1, dist1, kernel_func, sigma)
    ex_x_y_from_q_of_k = expected_discrepancy(dist2, dist2, kernel_func, sigma)
    ex_x_from_p_y_from_q_of_k = expected_discrepancy(dist1, dist2, kernel_func, sigma)

    mmd_squared = ex_x_y_from_p_of_k + ex_x_y_from_q_of_k - 2 * ex_x_from_p_y_from_q_of_k
    return mmd_squared


def test():
    """Test function duplicated from paper's original code to ensure we get
    identical MMD results."""
    s1 = np.array([0.2, 0.8])
    s2 = np.array([0.3, 0.7])
    samples1 = [s1, s2]

    s3 = np.array([0.25, 0.75])
    s4 = np.array([0.35, 0.65])
    samples2 = [s3, s4]

    s5 = np.array([0.8, 0.2])
    s6 = np.array([0.7, 0.3])
    samples3 = [s5, s6]

    print('value of gaussian kernel function for s1 and s2:', gaussian_kernel(s1, s2))
    print('value of gaussian kernel function for s2 and s3:', gaussian_kernel(s2, s3))
    print('value of gaussian kernel function for s3 and s4:', gaussian_kernel(s3, s4))
    print('value of gaussian kernel function for s1 and s4:', gaussian_kernel(s1, s4))
    print('value of gaussian kernel function for s2 and s4:', gaussian_kernel(s2, s4))

    print('value of expected discrepancy function for samples1 and samples2:',
          expected_discrepancy(samples1, samples2, gaussian_kernel))
    print('value of expected discrepancy function for samples2 and samples3:',
          expected_discrepancy(samples2, samples3, gaussian_kernel))
    print('value of expected discrepancy function for samples1 and samples3:',
          expected_discrepancy(samples1, samples3, gaussian_kernel))

    print('MMD between samples1 and samples2: ', mmd(samples1, samples2))
    print('MMD between samples1 and samples3: ', mmd(samples1, samples3))
    print('MMD between samples2 and samples3: ', mmd(samples2, samples3))

# test()
