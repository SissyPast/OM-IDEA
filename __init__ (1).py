from functools import partial

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.lax import scan

from om_idea.utilities.jax_component import JaxComponent
from om_idea.utilities.timeseries import compute_yearly_growth_rate


class FleetEvolutionModel(JaxComponent):
    def initialize(self):
        self.options.declare("start_year")
        self.options.declare("end_year")
        self.options.declare("indexed_reference_fuel_intensity")

        self.options.declare("aircraft_types_table")
        self.options.declare("aircraft_asm_shares_table")
        self.options.declare("deliveries_table")
        self.options.declare("replacement_table")
        self.options.declare("retirement_curves_table")

    def setup(self):
        # Initialize model
        (
            self.normalized_fuel_intensity_per_aircraft,
            self.survival_curves,
            self.replacement_matrix,
            self.initial_distribution,
        ) = initialize_fleet_evolution_model(
            self.options["start_year"],
            self.options["end_year"],
            self.options["aircraft_types_table"],
            self.options["aircraft_asm_shares_table"],
            self.options["deliveries_table"],
            self.options["replacement_table"],
            self.options["retirement_curves_table"],
        )

        # Compute or retrieve dimensions
        num_years = self.options["end_year"] - self.options["start_year"] + 1
        num_aircraft = self.initial_distribution.shape[0]
        num_age_bins = self.initial_distribution.shape[1]

        # Inputs
        self.add_input("domestic_asm", shape=(num_years,))
        self.add_input("international_asm", shape=(num_years,))
        self.add_input("indexed_fuel_intensity_operations", shape=(num_years,))

        # Outputs
        self.add_output(
            "age_distribution_by_year", shape=(num_years, num_aircraft, num_age_bins)
        )
        self.add_output("indexed_fuel_intensity_technology", shape=(num_years))
        self.add_output("indexed_fuel_intensity", shape=(num_years))
        self.add_output("relative_fuel_intensity", shape=(num_years))

    def jax_compute(self, inputs):
        num_years = self.options["end_year"] - self.options["start_year"] + 1
        (
            age_distribution_by_year,
            indexed_fuel_intensity_technology,
            indexed_fuel_intensity,
            relative_fuel_intensity,
        ) = run_fleet_evolution_model(
            num_years,
            self.initial_distribution,
            self.survival_curves,
            inputs["domestic_asm"],
            inputs["international_asm"],
            self.replacement_matrix,
            self.normalized_fuel_intensity_per_aircraft,
            inputs["indexed_fuel_intensity_operations"],
            self.options["indexed_reference_fuel_intensity"],
        )

        return {
            "age_distribution_by_year": age_distribution_by_year,
            "indexed_fuel_intensity_technology": indexed_fuel_intensity_technology,
            "indexed_fuel_intensity": indexed_fuel_intensity,
            "relative_fuel_intensity": relative_fuel_intensity,
        }


# find sequential replacements and merge into the dicts
def _find_next_replacement(key, datedict, repdict):
    lastdate = max(datedict.keys())
    lastrep = datedict[lastdate]
    if lastrep != key:
        nextrepdict = _find_next_replacement(lastrep, repdict[lastrep], repdict)
        datedict = dict(list(datedict.items()) + list(nextrepdict.items()))
    return datedict


def _initialize_replacement_matrix(
    start_year,
    end_year,
    aircraft_types_dict,
    aircraft_name_to_code_map,
    replacement_table,
    surviving_types_index,
):
    """TODO: rewrite this more clearly"""

    # Initialize replacement matrix
    replacement_matrix = np.zeros(
        (end_year - start_year, len(aircraft_types_dict), len(aircraft_types_dict))
    )

    # Replacement dictionary
    replacement_dict = {}
    for code in set(replacement_table["Code"]):
        replacement_dict[code] = {}
        rows = replacement_table[replacement_table["Code"] == code]
        for replacement, year in zip(
            rows["Replace with"].tolist(), rows["Starting"].tolist()
        ):
            replacement_dict[code][year] = aircraft_name_to_code_map[replacement]

    repramptime = 4.0
    for key in replacement_dict.keys():
        replacement_dict[key] = _find_next_replacement(
            key, replacement_dict[key], replacement_dict
        )

        repyearkeyidx = 0
        repyears = list(replacement_dict[key].keys())
        repyears.sort()
        repyearlen = len(repyears)
        startrepyear = repyears[repyearkeyidx]
        if (repyearkeyidx + 1) > (repyearlen - 1):
            nextrepyear = 2100
        else:
            repyearkeyidx += 1
            nextrepyear = repyears[repyearkeyidx]
        repcode = replacement_dict[key][startrepyear]
        yeariter = iter(range(end_year - start_year))
        # if key == 441:
        # pdb.set_trace()
        for yearidx in yeariter:
            if (yearidx + start_year) >= nextrepyear:
                startrepyear = nextrepyear
                if (repyearkeyidx + 1) > (repyearlen - 1):
                    nextrepyear = 2100
                else:
                    repyearkeyidx += 1
                    nextrepyear = repyears[repyearkeyidx]
                # set prior replacement to ramp fractions
                for i in range(round(repramptime)):
                    replacement_matrix[
                        yearidx + i,
                        surviving_types_index[repcode],
                        surviving_types_index[key],
                    ] = 1 - (1.0 / repramptime) * (i + 1)
                repcode = replacement_dict[key][startrepyear]
                # set new replacement to ramp fractions
                for i in range(round(repramptime)):
                    replacement_matrix[
                        yearidx + i,
                        surviving_types_index[repcode],
                        surviving_types_index[key],
                    ] = (1.0 / repramptime) * (i + 1)
                # skip ramptime ahead
                for i in range(round(repramptime) - 1):
                    next(yeariter)
                continue
            replacement_matrix[
                yearidx, surviving_types_index[repcode], surviving_types_index[key]
            ] = 1.0

    return replacement_matrix


def _initialize_survival_curves(
    deliveries_table,
    start_year,
    aircraft_types_dict,
    reversed_aircraft_types_dict,
    aircraft_types_asm_shares,
    retirement_curves,
):
    """TODO: rewrite this more clearly"""

    # Turn the initial deliveries table into a matrix with aircraft types as rows,
    # columns as years, and number of surviving aircraft as value
    surviving = pd.pivot_table(
        deliveries_table[
            (deliveries_table["year"] != -1) & (deliveries_table["year"] <= start_year)
        ],
        values="surviving",
        index="type",
        columns="year",
        aggfunc="sum",
        fill_value=0,
    )

    # Reorder columns and get array
    # TODO: check the new order to change `surviving.columns[::-1]` into something more
    # meaningful (make the intended order clearer)
    surviving_matrix = surviving[surviving.columns[::-1]].to_numpy()

    # add one year of zeros at the end
    # surviving_matrix = np.column_stack((surviving_matrix,np.zeros(np.shape(surviving_matrix)[0])))
    surviving_types = surviving.index.tolist()

    # if aircraft is not in surviving list add to list and empty row in matrix
    for actype in aircraft_types_dict.Description:
        if actype not in surviving_types:
            surviving_types.append(actype)
            surviving_matrix = np.vstack(
                (surviving_matrix, np.zeros(np.shape(surviving_matrix)[1]))
            )

    surviving_types_index = {}
    for i, actype in enumerate(surviving_types):
        surviving_types_index[reversed_aircraft_types_dict[actype]] = i

    # convert from number of aircraft ops to relative asm
    initial_distribution = np.copy(surviving_matrix)
    percents = aircraft_types_asm_shares / aircraft_types_asm_shares.sum()
    factors = percents.ASM  # /percents.DEPARTURES_PERFORMED
    for i, code in enumerate(aircraft_types_asm_shares.Code):
        initial_distribution[surviving_types_index[code], :] = (
            surviving_matrix[surviving_types_index[code], :] * factors[i]
        )

    # get efficiencies
    relative_aircraft_fuel_intensities = [
        aircraft_types_dict[
            aircraft_types_dict.Code == reversed_aircraft_types_dict[ac]
        ].releff.values
        for ac in surviving_types
    ]
    relative_aircraft_fuel_intensities = np.array(
        relative_aircraft_fuel_intensities
    ).flatten()

    # load ret curves
    num_age_bins = len(surviving_matrix[1])
    survival_curves = np.tile(
        np.array(retirement_curves["Parametric Narrow Body"][:num_age_bins]).T,
        (len(surviving_matrix), 1),
    )

    return (
        relative_aircraft_fuel_intensities,
        survival_curves,
        surviving_types_index,
        initial_distribution,
    )


def initialize_fleet_evolution_model(
    start_year,
    end_year,
    aircraft_types_table,
    aircraft_asm_shares_table,
    deliveries_table,
    replacement_table,
    retirement_curves_table,
):
    # load types data
    # aircraft_types_dict = aircraft_types_table.set_index("Code").to_dict()[
    #     "Description"
    # ]
    reversed_aircraft_types_dict = aircraft_types_table.set_index(
        "Description"
    ).to_dict()["Code"]

    (
        relative_aircraft_fuel_intensities,
        survival_curves,
        surviving_types_index,
        initial_distribution,
    ) = _initialize_survival_curves(
        deliveries_table,
        start_year,
        aircraft_types_table,
        reversed_aircraft_types_dict,
        aircraft_asm_shares_table,
        retirement_curves_table,
    )

    replacement_matrix = _initialize_replacement_matrix(
        start_year,
        end_year,
        aircraft_types_table,
        reversed_aircraft_types_dict,
        replacement_table,
        surviving_types_index,
    )

    return (
        relative_aircraft_fuel_intensities,
        survival_curves,
        replacement_matrix,
        initial_distribution,
    )


def _compute_average_fuel_intensity(
    age_distribution_by_year,
    normalized_fuel_intensity_per_aircraft,
    indexed_fuel_intensity_operations,
    indexed_reference_fuel_intensity,
):
    """
    `age_distribution_by_year` is a 3d array of dimensions num_years x num_aircraft x num_age_bins
    """
    num_each_aircraft_per_year = jnp.sum(age_distribution_by_year, axis=2)
    num_all_aircraft_per_year = jnp.sum(age_distribution_by_year, axis=(1, 2))

    normalized_fuel_intensity_technology = (
        num_each_aircraft_per_year
        @ normalized_fuel_intensity_per_aircraft
        / num_all_aircraft_per_year
    )

    # Forcibly indexing the fuel intensity improvements due to technology improvements
    indexed_fuel_intensity_technology = (
        normalized_fuel_intensity_technology / normalized_fuel_intensity_technology[0]
    )

    # Applying fuel intensity improvements due to operational improvements
    indexed_fuel_intensity = (
        indexed_fuel_intensity_technology * indexed_fuel_intensity_operations
    )

    # Relative to reference intensity
    relative_fuel_intensity = indexed_fuel_intensity / indexed_reference_fuel_intensity

    return (
        indexed_fuel_intensity_technology,
        indexed_fuel_intensity,
        relative_fuel_intensity,
    )


def _fleet_evolution_yearly_step(
    survival_curves, previous_years_age_distribution, fleet_evolution_data
):
    # Retrieving input data for this year - fleet growth and replacement matrix
    this_years_fleet_growth = fleet_evolution_data["yearly_fleet_growth"]
    this_years_replacement_matrix = fleet_evolution_data["replacement_matrix"]

    # Retirements - shape num_aircraft x (num_age_bins - 1)
    retirements = previous_years_age_distribution[:, :-1] * (
        1 - survival_curves[:, 1:] / (survival_curves[:, :-1] + 1e-6)
    )

    # Subtract retired aircraft from histograms
    distribution_after_retirements = (
        previous_years_age_distribution[:, :-1] - retirements
    )

    # Required replacements include retirements that year plus everything that
    # exceeded the maximum age and "fell off" the upper bound histogram when
    # shifting it by one year - shape is simply (num_aircraft,)
    required_replacements = (
        jnp.sum(retirements, axis=1) + previous_years_age_distribution[:, -1]
    )

    # Growth of previous year's fleet - shape is simply (num_aircraft,)
    required_growth = (
        jnp.sum(previous_years_age_distribution[:, :], axis=1)
        * this_years_fleet_growth
        / 100
    )

    # Newly introduced aircraft (age 0) come from 1) replacements and 2) growth
    newly_introduced_aircraft = this_years_replacement_matrix @ (
        required_replacements + required_growth
    )

    # Assemble the full distribution for this year
    this_years_age_distribution = jnp.concatenate(
        (newly_introduced_aircraft[:, jnp.newaxis], distribution_after_retirements),
        axis=1,
    )

    # It needs to be returned twice because of jax.lax.scan - we both need to pass it to
    # the next loop iteration and store it in the results
    return this_years_age_distribution, this_years_age_distribution


def run_fleet_evolution_model(
    num_years,
    initial_age_distribution,
    survival_curves,
    domestic_asm,
    international_asm,
    replacement_matrix,
    normalized_fuel_intensity_per_aircraft,
    indexed_fuel_intensity_operations,
    indexed_reference_fuel_intensity,
):
    """Shapes:
    - age_distribution_by_year: num_years x num_aircraft x num_age_bins
    - survival_curves: num_aircraft x num_age_bins
    - yearly_fleet_growth: num_years
    - replacement_matrix: num_years x num_aircraft x num_aircraft
    """

    # Compute the yearly fleet growth based on current ASM evolution
    total_asm = domestic_asm + international_asm
    yearly_fleet_growth = compute_yearly_growth_rate(total_asm)

    # This is a for loop implemented efficiently with JAX
    # Note that the loop runs from index 1 (not 0) to index (num_forecast_years - 1)
    _, propagated_age_distribution_by_year = scan(
        partial(_fleet_evolution_yearly_step, survival_curves),
        initial_age_distribution,
        {
            "yearly_fleet_growth": yearly_fleet_growth,
            "replacement_matrix": replacement_matrix,
        },
        length=num_years - 1,
    )

    # Concatenate the initial year and the propagated years
    age_distribution_by_year = jnp.concatenate(
        (
            initial_age_distribution[jnp.newaxis, :, :],
            propagated_age_distribution_by_year,
        ),
        axis=0,
    )

    # Compute the relative yearly overall efficiency based on the new distribution
    (
        indexed_fuel_intensity_technology,
        indexed_fuel_intensity,
        relative_fuel_intensity,
    ) = _compute_average_fuel_intensity(
        age_distribution_by_year,
        normalized_fuel_intensity_per_aircraft,
        indexed_fuel_intensity_operations,
        indexed_reference_fuel_intensity,
    )

    return (
        age_distribution_by_year,
        indexed_fuel_intensity_technology,
        indexed_fuel_intensity,
        relative_fuel_intensity,
    )
