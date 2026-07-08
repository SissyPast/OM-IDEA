from .namespace import IdeaInputs


def get_inputs_shapes(
    num_years,
    num_types,
    num_missions,
    num_domestic_missions,
    num_international_missions,
    num_age_bins,
    num_energy_carriers,
):
    assert num_missions == num_domestic_missions + num_international_missions

    return {
        #
        # = Economics
        # == Domestic
        IdeaInputs.indexed_domestic_gdp: (num_years,),
        IdeaInputs.indexed_domestic_population: (num_years,),
        # == International
        IdeaInputs.indexed_atlantic_gdppc: (num_years,),
        IdeaInputs.indexed_pacific_gdppc: (num_years,),
        IdeaInputs.indexed_latam_gdppc: (num_years,),
        # == Energy prices
        IdeaInputs.energy_price_per_energy_source: (num_years, num_energy_carriers),
        #
        # = Aviation Economics
        # == Pass-Through
        IdeaInputs.pass_through_fraction: (num_years,),
        # == Load factors
        IdeaInputs.domestic_load_factor: (num_years,),
        IdeaInputs.international_load_factor: (num_years,),
        #
        #
        # = Missions
        IdeaInputs.domestic_missions_fractions: (num_years, num_domestic_missions),
        IdeaInputs.international_missions_fractions: (
            num_years,
            num_international_missions,
        ),
        #
        # = Fleet Evolution
        IdeaInputs.initial_asm_fractions: (num_types, num_missions, num_age_bins),
        IdeaInputs.retirement_curves: (num_types, num_age_bins),
        IdeaInputs.replacement_matrix: (num_years, num_missions, num_types, num_types),
        IdeaInputs.normalized_energy_intensity_per_aircraft: (num_types,),
        IdeaInputs.indexed_energy_intensity_changes_operations: (num_years,),
        #
        # = Type-Level Data
        IdeaInputs.energy_carrier_breakdown_per_type: (num_types, num_energy_carriers),
    }
