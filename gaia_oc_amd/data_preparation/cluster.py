import numpy as np
from astropy.coordinates import SkyCoord

from gaia_oc_amd.data_preparation.candidate_selection import isochrone_candidate_condition
from gaia_oc_amd.data_preparation.features import radius_feature_function


def most_likely_value(means, errors):
    """Calculates the most probable value of quantity for which there are multiple measurements
    and each measurement has its own uncertainty. This gives a better estimate
    than simply calculating the mean.

    Args:
        means (float, array): Mean values
        errors (float, array): Uncertainty values

    Returns:
        The most likely value

    """
    return np.sum(means / errors ** 2) / np.sum(1 / errors ** 2)


def isochrone_deltas(members, isochrone, colour='g_rp', delta_c0=0.3, delta_g0=2.4, member_fraction=0.90, alpha=0.95,
                     c_margin=0.1, g_margin=0.8):
    """Calculates the maximum separation in the colour and magnitude dimensions. This is done by iteratively decreasing
    some initial delta values until only a certain fraction of the members is selected by the
    isochrone condition as candidate. Some constant margins can be added to prevent the delta from being too strict.

    Args:
        members (pd.Dataframe): Dataframe containing members of the open cluster
        isochrone (Dataframe): Dataframe containing the colour and magnitude of the isochrone.
        colour (str): Which colour field to use ('bp_rp', 'g_rp')
        delta_c0 (float): Initial value for the colour delta
        delta_g0 (float): Initial value for the magnitude delta
        member_fraction (float): Fraction of the members to be selected as candidate by the isochrone condition
        alpha (float): Factor with which to scale the delta values to iteratively find the minimum
        c_margin (float): Margin added to colour delta
        g_margin (float): Margin added to magnitude delta

    Returns:
        delta_c (float): Maximum separation in colour
        delta_g (float): Maximum separation in magnitude

    """
    n_members = len(members)

    iso_cond = isochrone_candidate_condition(isochrone, delta_c0, delta_g0, colour=colour)
    member_candidate_labels = members.apply(iso_cond, axis=1).to_numpy()

    delta_c_member_fraction = delta_c0
    delta_g_member_fraction = delta_g0

    # Decrease the isochrone deltas until only a fraction of the members are selected as candidate
    while sum(member_candidate_labels) > member_fraction * n_members:
        delta_c_member_fraction *= alpha
        delta_g_member_fraction *= alpha
        iso_cond = isochrone_candidate_condition(isochrone, delta_c_member_fraction, delta_g_member_fraction,
                                                 colour=colour)
        member_candidate_labels = members.apply(iso_cond, axis=1).to_numpy()

    delta_c = delta_c_member_fraction / alpha + c_margin
    delta_g = delta_g_member_fraction / alpha + g_margin

    return delta_c, delta_g


def plx_deltas(members, cluster_parallax, cluster_ra, cluster_dec, cluster_dist, mean_plx_e, zpt_error,
               member_fraction=0.90, alpha=0.95, r_max_margin=15.):
    """Calculates the maximum separation in parallax, which consist of a value for sources closer than the cluster
    and a value for sources farther away. This is done by iteratively decreasing a radius until we obtain a certain
    fraction of the members within this radius. We add some constant margins to prevent the delta from being too strict.

    Args:
        members (pd.Dataframe): Dataframe containing members of the open cluster
        cluster_parallax (float): Mean parallax of the cluster members
        cluster_ra (float): Mean right ascension of the cluster members
        cluster_dec (float): Mean declination of the cluster members
        cluster_dist (float): Distance to the cluster
        mean_plx_e (float): Uncertainty in the mean of the parallax of the members
        zpt_error (float): Uncertainty in the parallax zero point
        member_fraction (float): Fraction of the members to be selected as candidate by the isochrone condition
        alpha (float): Factor with which to scale the delta values to iteratively find the minimum (must be smaller
            than 1)
        r_max_margin (float): Margin added to the radius enclosing a fraction of the members in sky position

    Returns:
        delta_plx_plus (float): Maximum separation in parallax for sources farther away than the cluster
        delta_plx_min (float): Maximum separation in parallax for sources closer than the cluster

    """
    n_members = len(members)
    r = members.apply(radius_feature_function(cluster_ra, cluster_dec, cluster_dist), axis=1).to_numpy()

    r_threshold = r.max()

    # Decrease the threshold radius until only a fraction of the members have a smaller radius
    while np.sum(r < r_threshold) > member_fraction * n_members:
        r_threshold *= alpha

    r_threshold = r_threshold / alpha + r_max_margin

    delta_plx0 = 3 * mean_plx_e + 3 * zpt_error
    delta_plx_plus = abs(cluster_parallax - 1000 / (1000 / cluster_parallax + r_threshold)) + delta_plx0
    delta_plx_min = abs(cluster_parallax - 1000 / (1000 / cluster_parallax - r_threshold)) + delta_plx0

    return delta_plx_plus, delta_plx_min


class Cluster:
    """A class designed to contain all relevant information about an open cluster.

    Args:
        cluster_parameters (dict): Dictionary containing cluster properties.
    """
    def __init__(self, cluster_parameters=None):
        self.name = None

        self.ra = None
        self.dec = None
        self.l = None
        self.b = None

        self.pmra = None
        self.pmdec = None
        self.parallax = None

        self.pmra_error = None
        self.pmdec_error = None
        self.parallax_error = None

        self.age = None
        self.a0 = None
        self.dist = None

        self.delta_pm = None
        self.delta_plx_plus = None
        self.delta_plx_min = None
        self.delta_c = None
        self.delta_g = None

        self.isochrone = None
        self.isochrone_colour = 'g_rp'

        self.source_error_weight = None

        self.train_members_label = None
        self.comparison_members_label = None

        if cluster_parameters is not None:
            self.set_parameters(cluster_parameters)
            if self.ra is not None and self.dec is not None:
                self.set_galactic_coordinates()

    def set_parameters(self, cluster_parameters):
        for param in cluster_parameters:
            setattr(self, param, cluster_parameters[param])

    def set_galactic_coordinates(self):
        gal_coord = SkyCoord(self.ra, self.dec, frame='icrs', unit='deg').galactic
        self.l = gal_coord.l.value
        self.b = gal_coord.b.value

    def set_train_members_label(self, label):
        self.train_members_label = label

    def set_comparison_members_label(self, label):
        self.comparison_members_label = label

    def update_astrometric_parameters(self, members):
        """Function that updates the astrometric parameters of the cluster based on a set of its members.
        This function is useful when the properties of the provided members are more precise
        than those of the members on which the current cluster properties are based.

        Args:
            members (pd.Dataframe): Dataframe containing members of the open cluster

        """
        # Recalculate mean values from the (training) members
        self.ra = members['ra'].mean()
        self.dec = members['dec'].mean()

        # Derive galactic coordinates from icrs coordinates
        self.set_galactic_coordinates()

        # Calculate the most probable proper motion and parallax of the cluster
        for param in ['pmra', 'pmdec', 'parallax']:
            setattr(self, param, most_likely_value(members[param], members[f'{param}_error']))

        self.pmra_error = np.std(members['pmra'])
        self.pmdec_error = np.std(members['pmdec'])
        self.parallax_error = np.std(members['parallax'])

    def set_candidate_selection_parameters(self, members, isochrone, colour='g_rp', source_error_weight=3.,
                                           pm_error_weight=3., r_max_margin=15., zpt_error=0.015, delta_c0=0.3,
                                           delta_g0=2.4, c_margin=0.1, g_margin=0.8, member_fraction=0.90,
                                           alpha=0.95):
        """Function that sets the parameters which are used in the candidate selection. These include the
        maximum separation delta which define the zero-error boundaries between candidates and non-members.

        Args:
            members (pd.Dataframe): Dataframe containing members of the open cluster
            isochrone (Dataframe): Dataframe containing the colour and magnitude of the isochrone.
            colour (str): Which colour field to use ('bp_rp', 'g_rp')
            source_error_weight (float): The number of sigma a candidate may deviate from zero-error boundary
            pm_error_weight (float): The weight of the standard deviation in the cluster member
                proper motion in the definition of the zero-error boundary
            r_max_margin (float): Margin added to the maximum radius used for the parallax delta
            zpt_error (float): Uncertainty in the parallax zero point
            delta_c0 (float): Initial value for the colour delta
            delta_g0 (float): Initial value for the magnitude delta
            c_margin (float): Margin added to colour delta
            g_margin (float): Margin added to magnitude delta
            member_fraction (float): Fraction of the members to be selected as candidate by the isochrone condition
            alpha (float): Factor with which to scale the delta values to iteratively find the minimum

        """
        self.isochrone = isochrone
        self.isochrone_colour = colour
        self.source_error_weight = source_error_weight

        # Uncertainties in the mean astrometric variables
        mean_pmra_error = np.sqrt(1 / np.sum(1 / members['pmra_error'] ** 2))
        mean_pmdec_error = np.sqrt(1 / np.sum(1 / members['pmdec_error'] ** 2))
        mean_plx_error = np.sqrt(1 / np.sum(1 / members['parallax_error'] ** 2))

        # Calculate the various maximum separations
        self.delta_pm = np.sqrt((pm_error_weight * self.pmra_error + 3 * mean_pmra_error)**2 +
                                (pm_error_weight * self.pmdec_error + 3 * mean_pmdec_error)**2)
        self.delta_plx_plus, self.delta_plx_min = plx_deltas(members, self.parallax, self.ra, self.dec, self.dist,
                                                             mean_plx_error, zpt_error, r_max_margin=r_max_margin,
                                                             member_fraction=member_fraction, alpha=alpha)
        self.delta_c, self.delta_g = isochrone_deltas(members, isochrone, colour=colour,
                                                      delta_c0=delta_c0, delta_g0=delta_g0,
                                                      c_margin=c_margin, g_margin=g_margin,
                                                      member_fraction=member_fraction, alpha=alpha)
