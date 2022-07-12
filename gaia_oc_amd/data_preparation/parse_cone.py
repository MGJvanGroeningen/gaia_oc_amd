from gaia_oc_amd.data_preparation.sets import member_set, candidate_and_non_members_set
from gaia_oc_amd.data_preparation.features import Features
from gaia_oc_amd.data_preparation.utils import add_columns


def parse_cone(cone, cluster, isochrone, member_ids, member_probs=None, comparison_ids=None, comparison_probs=None,
               max_r=60., pm_error_weight=5., g_delta=1.5, bp_rp_delta=0.5, source_error_weight=3.):
    """Parses the sources in the cone search into member, candidate, non-member and comparison sources.
    Members are obtained through the provided member source identities. Sources are selected as candidates if
    they satisfy conditions in proper motion, parallax, colour and magnitude.

    Args:
        cone (Dataframe): Dataframe containing the data of the sources in the cone search.
        cluster (Cluster): Cluster object
        isochrone (Dataframe): Dataframe containing colour and magnitude values of the isochrone curve.
        member_ids (Series, array): Source identities of the members
        member_probs (Series, array): Membership probabilities of the members
        comparison_ids (Series, array): Source identities of the comparison members
        comparison_probs (Series, array): Membership probabilities of the comparison members
        max_r (float): Maximum candidate radius in parsec which determines the maximum deviation in parallax
            from the cluster mean
        pm_error_weight (float): Maximum candidate deviation in proper motion (in number of sigma)
            from the cluster mean
        g_delta (float): Maximum candidate deviation in G magnitude from the isochrone
        bp_rp_delta (float): Maximum candidate deviation in colour from the isochrone
        source_error_weight (float): How many sigma candidates may lie outside the maximum deviations.

    Returns:
        members (Dataframe): Dataframe containing member sources
        candidates (Dataframe): Dataframe containing candidate sources
        non_members (Dataframe): Dataframe containing non_member sources
        comparison (Dataframe): Dataframe containing comparison sources

    """
    # Construct the member set
    members = member_set(cone, member_ids, member_probs)

    # Update the cluster parameters based on the members
    cluster.update_parameters(members)

    # Set cluster parameters that are relevant for the candidate selection and training features
    cluster.set_feature_parameters(members, max_r, pm_error_weight, g_delta, bp_rp_delta, source_error_weight)

    # Construct the candidate and non-member set
    candidates, non_members = candidate_and_non_members_set(cone, cluster, isochrone)

    # Remove sources from the members if they were selected as non_members
    members = members[~members['source_id'].isin(non_members['source_id'])]

    # Add the custom training features to the columns of the source dataframes
    training_features = Features(cluster, isochrone)
    add_columns([members, candidates, non_members], training_features.functions, training_features.labels)

    # Optionally construct the comparison set
    if comparison_ids is not None:
        comparison = member_set(cone, comparison_ids, comparison_probs)
    else:
        comparison = None

    return members, candidates, non_members, comparison
