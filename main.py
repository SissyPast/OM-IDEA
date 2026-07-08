from enum import StrEnum, auto
from typing import NamedTuple

import numpy as np


class IdeaOptions(NamedTuple):
    # Global Options
    # --------------

    # Number of forecast years - length of the majority of the vectors we work with
    num_forecast_years: int

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

    indexed_reference_fuel_price: np.ndarray
    reference_fuel_costs_fraction_of_operating_costs: float

    # Reference fuel intensity baked in reference projections
    # Reference RPM projection corresponds to this fuel intensity projection, so a change in
    # efficiency affecting RPM must be taken with respect to this reference intensity
    initial_fleet_level_fuel_intensity: float
    indexed_reference_fleet_level_fuel_intensity: np.ndarray

    # Load factor
    reference_domestic_load_factor: np.ndarray
    reference_international_load_factor: np.ndarray

    reference_business_rpm_fraction: float
    reference_shorthaul_rpm_fraction: float
    domestic_price_elasticities: dict[str, dict[str, float]]
    international_price_elasticities: dict[str, dict[str, float]]

    start_year: int
    end_year: int
    aircraft_types_table: np.ndarray
    aircraft_asm_shares_table: np.ndarray
    deliveries_table: np.ndarray
    replacement_table: np.ndarray
    retirement_curves_table: np.ndarray


class IdeaInputs(StrEnum):
    indexed_domestic_gdp = auto()
    indexed_domestic_population = auto()

    indexed_atlantic_gdppc = auto()
    indexed_pacific_gdppc = auto()
    indexed_latam_gdppc = auto()

    indexed_fuel_price = auto()

    pass_through_fraction = auto()

    domestic_load_factor = auto()
    international_load_factor = auto()

    indexed_fleet_level_fuel_intensity_operations = auto()


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

    # Fuel intensity
    age_distribution_by_year = auto()
    indexed_fleet_level_fuel_intensity_technology = auto()
    indexed_fleet_level_fuel_intensity = auto()
    relative_fleet_level_fuel_intensity = auto()

    # Fleet-level quantities
    fuel_quantity = auto()
    energy_quantity = auto()
    carbon_dioxide_quantity = auto()
