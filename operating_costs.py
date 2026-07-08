from functools import partial

import jax.numpy as jnp
from jax.lax import scan

from om_idea.utilities.jax_component import JaxComponent

"""
## Inputs:

    - Initial ASM fractions
        - Describes the breakdown of ASM per type, mission, and age
        - Size num_types x num_missions x num_age_bins
        - Sums to 1
    - Retirement curves
        - For each type, gives the fraction of ASMs getting retired as a function of age
        - Size num_types x num_age_bins
    - Relative energy intensities
        - For each type, the energy intensity relative to today's aircraft
        - Size num_types
    - Energy carrier breakdown
        - For each type, indicates the contribution of each energy carrier
        - Each row sums to 1
        - Size num_types x num_energy_carriers 
    - Relative energy intensity due to operations
        - For every year, provides the relative change in energy intensity due to operational improvements (1 means no change due to operations and values smaller than 1 mean a reduction in energy consumption)
        - Size num_years
    - Replacement matrix
        - For every year and mission, maps an original type (3rd dimension) to its replacement(s) (4th dimension)
        - Given a year and a mission, each row must sum to 1, meaning that 1 ASM gets replaced by 1 ASM, from potentially multiple types
        - Size num_years x num_missions x num_types x num_types
    

## Outputs:

    - ASM fractions per year
        - For every projected year, describes the breakdown of ASM per type, mission, and age
        - Size num_years x num_types x num_missions x num_age_bins
        - For every year, sums to 1
    - Average fuel intensity per energy carrier per year
        - For every projected year and every mission, provides the average energy intensity of the fleet
        - Size num_years x num_missions x num_energy_carriers

"""


class FleetEvolutionModel(JaxComponent):
    def initialize(self):
        self.options.declare("num_years")
        self.options.declare("num_energy_carriers")
        self.options.declare("num_types")
        self.options.declare("num_missions")
        self.options.declare("num_age_bins")

    def setup(self):
        # Options
        num_years = self.options["num_years"]
        num_energy_carriers = self.options["num_energy_carriers"]
        num_types = self.options["num_types"]
        num_missions = self.options["num_missions"]
        num_age_bins = self.options["num_age_bins"]

        # Inputs
        self.add_input(
            "initial_asm_fractions", shape=(num_types, num_missions, num_age_bins)
        )
        self.add_input("retirement_curves", shape=(num_types, num_age_bins))
        self.add_input("yearly_fleet_growth", shape=(num_years - 1, num_missions))
        self.add_input(
            "replacement_matrix", shape=(num_years, num_missions, num_types, num_types)
        )
        self.add_input("normalized_energy_intensity_per_aircraft", shape=(num_types,))
        self.add_input(
            "indexed_energy_intensity_changes_operations", shape=(num_years,)
        )
        self.add_input(
            "energy_carrier_breakdown_per_type",
            shape=(num_types, num_energy_carriers),
        )
        self.add_input(
            "energy_price_per_energy_source", shape=(num_years, num_energy_carriers)
        )

        # Outputs
        self.add_output(
            "asm_fractions_per_year",
            shape=(num_years, num_types, num_missions, num_age_bins),
        )
        self.add_output("indexed_energy_intensity_technology", shape=(num_years,))
        self.add_output("indexed_energy_intensity", shape=(num_years,))
        self.add_output(
            "yearly_fleet_energy_use_breakdown", shape=(num_years, num_energy_carriers)
        )
        self.add_output("yearly_average_energy_price", shape=(num_years,))
        self.add_output("indexed_energy_price", shape=(num_years,))

    def jax_compute(self, inputs):
        (
            asm_fractions_per_year,
            indexed_energy_intensity_technology,
            indexed_energy_intensity,
        ) = run_fleet_evolution_model(
            self.options["num_years"],
            inputs["initial_asm_fractions"],
            inputs["retirement_curves"],
            inputs["yearly_fleet_growth"],
            inputs["replacement_matrix"],
            inputs["normalized_energy_intensity_per_aircraft"],
            inputs["indexed_energy_intensity_changes_operations"],
        )

        energy_use_breakdown, average_energy_price, indexed_energy_price = energy_price(
            asm_fractions_per_year,
            inputs["energy_carrier_breakdown_per_type"],
            inputs["energy_price_per_energy_source"],
        )

        return {
            "asm_fractions_per_year": asm_fractions_per_year,
            "indexed_energy_intensity_technology": indexed_energy_intensity_technology,
            "indexed_energy_intensity": indexed_energy_intensity,
            "yearly_fleet_energy_use_breakdown": energy_use_breakdown,
            "yearly_average_energy_price": average_energy_price,
            "indexed_energy_price": indexed_energy_price,
        }


def compute_average_fuel_intensity(
    age_distribution_by_year,  # shape = (num_years, num_types, num_missions, num_age_bins)
    normalized_fuel_intensity_per_aircraft,  # shape = (num_types,)
    indexed_fuel_intensity_operations,
):
    """
    `age_distribution_by_year` is a 3d array of dimensions num_years x num_aircraft x num_age_bins
    """

    # Normalized fuel intensity - due to technology only
    normalized_fuel_intensity_technology = jnp.einsum(
        "ijkl,j->i",
        age_distribution_by_year,
        normalized_fuel_intensity_per_aircraft,
    )

    # Forcibly indexing the fuel intensity improvements due to technology improvements
    indexed_fuel_intensity_technology = (
        normalized_fuel_intensity_technology / normalized_fuel_intensity_technology[0]
    )

    # Applying fuel intensity improvements due to operational improvements
    indexed_fuel_intensity = (
        indexed_fuel_intensity_technology * indexed_fuel_intensity_operations
    )

    return indexed_fuel_intensity_technology, indexed_fuel_intensity


def energy_price(
    asm_fractions_per_year,
    energy_carrier_breakdown_per_type,
    energy_price_per_energy_source,
):
    energy_use_breakdown = jnp.einsum(
        "ijkl,jm->im",
        asm_fractions_per_year,
        energy_carrier_breakdown_per_type,
    )

    average_energy_price = jnp.einsum(
        "ij,ij->i",
        energy_use_breakdown,
        energy_price_per_energy_source,
    )

    indexed_energy_price = average_energy_price / average_energy_price[0]

    return energy_use_breakdown, average_energy_price, indexed_energy_price


def _fleet_evolution_yearly_step(
    retirement_curves,  # shape = (num_types, num_age_bins)
    previous_years_age_distribution,  # shape = (num_types, num_missions, num_age_bins)
    fleet_evolution_data,
):
    # Retrieving input data for this year - fleet growth and replacement matrix

    # shape = (num_missions,)
    this_years_fleet_growth = fleet_evolution_data["yearly_fleet_growth"]

    # shape = (num_types, num_missions, num_types)
    this_years_replacement_matrix = fleet_evolution_data["replacement_matrix"]

    # Retirements
    # shape = (num_types, num_missions, num_age_bins)
    retirements = jnp.einsum(
        "ijk,ik->ijk",
        previous_years_age_distribution,
        retirement_curves,
    )

    # Applying retirements and shifting age distribution by one year to the right
    # shape = (num_types, num_missions, num_age_bins)
    distribution_after_retirements = (previous_years_age_distribution - retirements)[
        ..., :-1
    ]

    # ASMs that need to be replaced for each type and mission
    # shape = (num_types, num_missions)
    required_replacements = jnp.sum(retirements, axis=2)

    # Growth of previous year's fleet
    # shape = (num_types, num_missions)
    required_growth = jnp.einsum(
        "ijk,j->ij",
        previous_years_age_distribution,
        this_years_fleet_growth,
    )

    # Newly introduced aircraft (age 0) come from 1) replacements and 2) growth
    #
    # required_replacements: shape = (num_types, num_missions,)
    # required_growth: shape = (num_types, num_missions,)
    # this_years_replacement_matrix: shape = (num_missions, num_types, num_types)
    #
    # newly_introduced_aircraft: shape = (num_types, num_missions)
    newly_introduced_aircraft = jnp.einsum(
        "ij,jik->kj",
        required_replacements + required_growth,
        this_years_replacement_matrix,
    )

    # Assemble the full distribution for this year
    this_years_age_distribution = jnp.concatenate(
        (newly_introduced_aircraft[..., jnp.newaxis], distribution_after_retirements),
        axis=-1,
    )

    # Normalize so that all fractions sum to 1
    this_years_age_distribution /= 1 + required_growth.sum()

    # It needs to be returned twice because of jax.lax.scan - we both need to pass it to
    # the next loop iteration and store it in the results
    return this_years_age_distribution, this_years_age_distribution


def run_fleet_evolution_model(
    num_years,
    initial_age_distribution,
    retirement_curves,
    yearly_fleet_growth,
    replacement_matrix,
    normalized_energy_intensity_per_aircraft,
    indexed_energy_intensity_operations,
):
    # This is a for loop implemented efficiently with JAX
    # Note that the loop runs from index 1 (not 0) to index (num_forecast_years - 1)
    _, propagated_age_distribution_by_year = scan(
        partial(_fleet_evolution_yearly_step, retirement_curves),
        initial_age_distribution,
        {
            "yearly_fleet_growth": yearly_fleet_growth,
            "replacement_matrix": replacement_matrix[:-1, :, :, :],
        },
        length=num_years - 1,
    )

    # Concatenate the initial year and the propagated years
    age_distribution_by_year = jnp.concatenate(
        (
            initial_age_distribution[jnp.newaxis, ...],
            propagated_age_distribution_by_year,
        ),
        axis=0,
    )

    # Compute the relative yearly overall efficiency based on the new distribution
    (indexed_energy_intensity_technology, indexed_energy_intensity) = (
        compute_average_fuel_intensity(
            age_distribution_by_year,
            normalized_energy_intensity_per_aircraft,
            indexed_energy_intensity_operations,
        )
    )

    return (
        age_distribution_by_year,
        indexed_energy_intensity_technology,
        indexed_energy_intensity,
    )
