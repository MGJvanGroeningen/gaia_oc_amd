import numpy as np
from astropy.coordinates import SkyCoord


def most_likely_value(values, errors):
    return np.sum(values / errors ** 2) / np.sum(1 / errors ** 2)


class Cluster:
    def __init__(self, cluster_parameters=None):
        self.name = None

        self.ra = None
        self.dec = None
        self.l = None
        self.b = None
        self.pmra = None
        self.pmra_error = None
        self.pmdec = None
        self.pmdec_error = None
        self.parallax = None
        self.parallax_error = None

        self.age = None
        self.a_v = None
        self.dist = None

        self.r_t = None

        self.std_of_pmra_mean = None
        self.std_of_pmdec_mean = None
        self.std_of_plx_mean = None

        self.pmra_delta = None
        self.pmdec_delta = None
        self.plx_delta_plus = None
        self.plx_delta_min = None

        self.g_delta = None
        self.bp_rp_delta = None
        self.source_errors = None

        if cluster_parameters is not None:
            for param in cluster_parameters:
                setattr(self, param, cluster_parameters[param])

    def update_parameters(self, members):
        for param in ['ra', 'dec', 'pmra', 'pmdec', 'parallax']:
            setattr(self, param, most_likely_value(members[param], members[f'{param}_error']))
        self.dist = 1000 / self.parallax

        gal_coord = SkyCoord(self.ra, self.dec, frame='icrs', unit='deg').galactic
        self.l = gal_coord.l.value
        self.b = gal_coord.b.value

        self.pmra_error = members['pmra'].std()
        self.pmdec_error = members['pmdec'].std()
        self.parallax_error = members['parallax'].std()

    def set_feature_parameters(self, members, max_r=60., pm_errors=5., g_delta=1.5, bp_rp_delta=0.5, source_errors=3.,
                               verbose=False):
        self.std_of_pmra_mean = np.sqrt(1 / np.sum(1 / members['pmra_error'] ** 2))
        self.std_of_pmdec_mean = np.sqrt(1 / np.sum(1 / members['pmdec_error'] ** 2))
        self.std_of_plx_mean = np.sqrt(1 / np.sum(1 / members['parallax_error'] ** 2))

        self.pmra_delta = 3 * self.std_of_pmra_mean + pm_errors * self.pmra_error
        self.pmdec_delta = 3 * self.std_of_pmdec_mean + pm_errors * self.pmdec_error

        max_plx_delta_plus = abs(self.parallax - 1000 / (1000 / self.parallax + max_r))
        max_plx_delta_min = abs(self.parallax - 1000 / (1000 / self.parallax - max_r))
        self.plx_delta_plus = 3 * self.std_of_plx_mean + max_plx_delta_plus
        self.plx_delta_min = 3 * self.std_of_plx_mean + max_plx_delta_min

        self.g_delta = g_delta
        self.bp_rp_delta = bp_rp_delta
        self.source_errors = source_errors

        if verbose:
            print('Standard deviations of the cluster mean')
            print('pmra:', self.std_of_pmra_mean, 'mas/yr')
            print('pmdec:', self.std_of_pmdec_mean, 'mas/yr')
            print('parallax:', self.std_of_plx_mean, 'mas')
            print(' ')

            print('Maximum separations')
            print('pmra:', self.pmra_delta, 'mas/yr')
            print('pmdec:', self.pmdec_delta, 'mas/yr')
            print('parallax +:', self.plx_delta_plus, 'mas')
            print('parallax -:', self.plx_delta_min, 'mas')
            print(' ')