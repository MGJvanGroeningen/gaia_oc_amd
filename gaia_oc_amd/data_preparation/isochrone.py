import numpy as np
import pandas as pd


def make_isochrone(isochrone_path, cluster):
    isochrones = pd.read_csv(isochrone_path,
                             names=['Zini', 'MH', 'logAge', 'Mini', 'int_IMF', 'Mass', 'logL', 'logTe', 'logg',
                                    'label', 'McoreTP', 'C_O', 'period0', 'period1', 'period2', 'period3',
                                    'period4', 'pmode', 'Mloss', 'tau1m', 'X', 'Y', 'Xc', 'Xn', 'Xo', 'Cexcess',
                                    'Z', 'mbolmag', 'Gmag',  'G_BPmag', 'G_RPmag'],
                             delim_whitespace=True,
                             comment='#')

    isochrones.rename(columns={'Gmag': 'phot_g_mean_mag'}, inplace=True)
    isochrones['bp_rp'] = isochrones['G_BPmag'] - isochrones['G_RPmag']

    ages = np.array(list(set(isochrones['logAge'].values)))
    closest_age = ages[np.argmin(np.abs(ages - cluster.age))]
    isochrone = isochrones[(isochrones['logAge'] == closest_age) & (isochrones['label'] <= 7)].copy()

    distance_modulus = 5 * np.log10(cluster.dist) - 5
    isochrone['phot_g_mean_mag'] += distance_modulus

    return isochrone
