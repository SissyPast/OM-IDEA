import itertools as it


def get_combinations(*lists):
    return list(it.product(*lists))


def flatten_price_elasticities(scope_distance_purpose_combinations, price_elasticities):
    return {
        f"{scope}_{distance}_{purpose}_price_sensitivity": price_elasticities[scope][
            purpose
        ][distance]
        for scope, distance, purpose in scope_distance_purpose_combinations
    }
