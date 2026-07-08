from enum import StrEnum, auto
from typing import NamedTuple

import numpy as np


class IdeaOptions(NamedTuple):
    # Problem dimensions
    # --------------

    # Number of forecast years - length of the majority of the vectors we work with
    num_years: int
    num_types: int
    num_missions: int
    num_domestic_missions: int
    num_international_missions: int
    num_age_bins: int
    num_energy_carriers: int

    # Initial demand scaling based on GDP per capita
    # -----------------------------------------------

    # Slope of the number of trips vs. GDP per capita data obtained by regressing Airbus's data
    num_trips_vs_gdppc_slope: float

    # Initial timeseries values
    initial_domestic_rpm: float
    initial_domestic_gdppc: float
    initial_domestic_num_trips_per_capita: float

    initial_atlantic_rpm: float
    initial_atlantic_gdppc: float
    initial_atlantic_num_trips_per_capita: float

    initial_pacific_rpm: float
    initial_pacific_gdppc: float
    initial_pacific_num_trips_per_capita: float

    initial_latam_rpm: float
    initial_latam_gdppc: float
    initial_latam_num_trips_per_capita: float

    initial_international_rpm: float

    # Indexed reference projections
    indexed_reference_domestic_rpm: np.ndarray
    indexed_reference_domestic_gdppc: np.ndarray

    indexed_reference_atlantic_rpm: np.ndarray
    indexed_reference_atlantic_gdppc: np.ndarray

    indexed_reference_pacific_rpm: np.ndarray
    indexed_reference_pacific_gdppc: np.ndarray

    indexed_reference_latam_rpm: np.ndarray
    indexed_reference_latam_gdppc: np.ndarray

    indexed_reference_international_rpm: np.ndarray

    # Main Fleet Efficiency Loop
    # --------------------------

    # Costs
    indexed_reference_energy_price: np.ndarray
    reference_energy_costs_fraction_of_operating_costs: float

    # Reference fuel intensity baked in reference projections
    # Reference RPM projection corresponds to this fuel intensity projection, so a change in
    # efficiency affecting RPM must be taken with respect to this reference intensity
    initial_fleet_level_energy_intensity: float
    indexed_reference_fleet_level_energy_intensity: np.ndarray

    # Load factor
    reference_domestic_load_factor: np.ndarray
    reference_international_load_factor: np.ndarray

    # Demand scaling
    reference_business_rpm_fraction: float
    reference_shorthaul_rpm_fraction: float
    domestic_price_elasticities: dict[str, dict[str, float]]
    international_price_elasticities: dict[str, dict[str, float]]

    # start_year: int
    # end_year: int
    # aircraft_types_table: np.ndarray
    # aircraft_asm_shares_table: np.ndarray
    # deliveries_table: np.ndarray
    # replacement_table: np.ndarray
    # retirement_curves_table: np.ndarray


class IdeaInputs(StrEnum):
    #
    # = Economics
    # == Domestic
    indexed_domestic_gdp = auto()
    indexed_domestic_population = auto()
    # == International
    indexed_atlantic_gdppc = auto()
    indexed_pacific_gdppc = auto()
    indexed_latam_gdppc = auto()
    # == Energy prices
    energy_price_per_energy_source = auto()
    #
    # = Aviation Economics
    # == Pass-Through
    pass_through_fraction = auto()
    # == Load factors
    domestic_load_factor = auto()
    international_load_factor = auto()
    #
    #
    # = Missions
    domestic_missions_fractions = auto()
    international_missions_fractions = auto()
    #
    # = Fleet Evolution
    initial_asm_fractions = auto()
    retirement_curves = auto()
    replacement_matrix = auto()
    normalized_energy_intensity_per_aircraft = auto()
    indexed_energy_intensity_changes_operations = auto()
    #
    # = Type-Level Data
    energy_carrier_breakdown_per_type = auto()


class IdeaCoupling(StrEnum):
    # Impact of GDP and population on domestic RPM
    indexed_domestic_gdppc = auto()
    relative_delta_domestic_rpm_gdppc = auto()

    # Impact of GDPPC on international RPM
    relative_delta_atlantic_rpm_gdppc = auto()
    relative_delta_pacific_rpm_gdppc = auto()
    relative_delta_latam_rpm_gdppc = auto()
    relative_delta_international_rpm_gdppc = auto()

    # Impact of fuel intensity and load factor on RPM
    relative_delta_operating_cost = auto()
    relative_delta_domestic_ticket_price = auto()
    relative_delta_international_ticket_price = auto()
    relative_delta_domestic_rpm_fuel_lf = auto()
    relative_delta_international_rpm_fuel_lf = auto()

    # Aggregating GDPPC/fuel/LF impacts
    relative_delta_domestic_rpm = auto()
    relative_delta_international_rpm = auto()

    # Updated RPM and ASM
    domestic_rpm = auto()
    domestic_asm = auto()
    international_rpm = auto()
    international_asm = auto()

    # Fleet
    asm_fractions_per_year = auto()
    yearly_fleet_growth = auto()

    # Energy intensity
    indexed_fleet_level_energy_intensity_technology = auto()
    indexed_fleet_level_energy_intensity = auto()
    relative_fleet_level_fuel_intensity = auto()

    # Energy price
    fleet_level_energy_use_breakdown = auto()
    fleet_level_energy_price = auto()
    indexed_fleet_level_energy_price = auto()

    # Fleet-level quantities
    # yearly_fleet_energy_use_breakdown = auto()
    fuel_quantity = auto()
    energy_quantity = auto()
    carbon_dioxide_quantity = auto()
