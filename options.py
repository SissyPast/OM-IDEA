from pathlib import Path

import pandas as pd

from om_idea.utilities.dataframe import df_columns_to_numpy_arrays
from om_idea.utilities.extrapolate import extrapolate

# Dictionary that maps new column names to the original FAA forecast CSV column names
FAA_FORECAST_CSV_COLUMNS_MAP = dict(
    # Years
    years="Years",
    # Domestic
    domestic_gdp="gdp",
    domestic_rpm="domrpm",
    domestic_asm="domasm",
    domestic_load_factor="domlf",
    domestic_seats="domesticseats",
    domestic_distance="domesticdist",
    domestic_fuel_quantity="domjetfuel",
    # International
    international_rpm="intrpm",
    international_asm="intasm",
    international_load_factor="intlf",
    international_fuel_quantity="intjetfuel",
    # Fuel price
    fuel_price="domjetfuelprice",
    # GDP
    regional_gdp_atlantic="gdpatlantic",
    regional_gdp_latam="gdplatam",
    regional_gdp_pacific="gdppacific",
    # Regional RPM
    regional_rpm_atlantic="atlanticrpm",
    regional_rpm_pacific="pacificrpm",
    regional_rpm_latam="latamrpm",
)


def load_faa_forecast(faa_forecast_filepath: Path) -> dict:
    # Read the FAA forecast
    faa_forecast_df = pd.read_csv(faa_forecast_filepath)

    # Convert to dict with better column names
    faa_forecast_dict = df_columns_to_numpy_arrays(
        faa_forecast_df, FAA_FORECAST_CSV_COLUMNS_MAP
    )

    # Make some alterations
    faa_forecast_dict["domestic_load_factor"] /= 100
    faa_forecast_dict["international_load_factor"] /= 100
    faa_forecast_dict["fuel_price"] /= 100

    # Bundle everything into a namedtuple
    return faa_forecast_dict


def load_us_population_data(us_population_forecast_filepath, years):
    """
    TODO Raphael:
        - there are a few cases where the following might break, need to make
          assumptions explicit
        - not sure about extrapolating after cropping...
    """
    # Load US population data
    population_data = pd.read_csv(us_population_forecast_filepath)

    # Extract what's of interest
    us_population_years = population_data.Year.to_numpy()
    us_population_count = population_data.Population.to_numpy()

    # Extrapolate population data to length of forecast
    us_population_count = extrapolate(years, us_population_years, us_population_count)

    return {"us_population_count": us_population_count}


AIRBUS_TO_FAA_REGIONS_MAP = {
    "North America": "domestic",
    "Asia Pacific": "pacific",
    "Lat.& Carib.": "latam",
    "Europe": "atlantic",
}


def load_airbus_demand_data(
    airbus_trips_vs_gdp_filepath, airbus_regional_data_filepath
):
    # Read data
    trip_vs_gdp_data = pd.read_csv(airbus_trips_vs_gdp_filepath)
    trip_vs_gdp_continent_data = pd.read_csv(
        airbus_regional_data_filepath, index_col=0
    ).to_dict(orient="index")

    # Organize it
    return {
        "trips_vs_gdp_data": {
            "gdp_per_capita": trip_vs_gdp_data["gdppercap"].to_numpy(),
            "num_trips": trip_vs_gdp_data["trips"].to_numpy(),
        },
        "regional_data": {
            faa_name: {
                "gdp_per_capita": trip_vs_gdp_continent_data[airbus_name]["gdppercap"],
                "num_trips": trip_vs_gdp_continent_data[airbus_name]["trips"],
            }
            for airbus_name, faa_name in AIRBUS_TO_FAA_REGIONS_MAP.items()
        },
    }


def load_fleet_data(
    aircraft_types_filepath,
    aircraft_asm_shares_filepath,
    deliveries_table_filepath,
    replacement_table_filepath,
    retirement_curves_table_filepath,
):
    return {
        "aircraft_types_table": pd.read_csv(aircraft_types_filepath),
        "aircraft_asm_shares_table": pd.read_csv(aircraft_asm_shares_filepath),
        "deliveries_table": pd.read_csv(deliveries_table_filepath),
        "replacement_table": pd.read_csv(replacement_table_filepath),
        "retirement_curves_table": pd.read_csv(retirement_curves_table_filepath),
    }


def load_data(
    faa_forecast_filepath,
    us_population_forecast_filepath,
    airbus_trips_vs_gdp_filepath,
    airbus_regional_data_filepath,
    aircraft_types_filepath,
    aircraft_asm_shares_filepath,
    deliveries_table_filepath,
    replacement_table_filepath,
    retirement_curves_table_filepath,
):
    # Load FAA forecast
    faa_forecast_data = load_faa_forecast(faa_forecast_filepath)

    # Load US population data
    us_population_data = load_us_population_data(
        us_population_forecast_filepath, faa_forecast_data["years"]
    )

    # Airbus demand data
    airbus_demand_data = load_airbus_demand_data(
        airbus_trips_vs_gdp_filepath, airbus_regional_data_filepath
    )

    # Fleet data
    fleet_data = load_fleet_data(
        aircraft_types_filepath,
        aircraft_asm_shares_filepath,
        deliveries_table_filepath,
        replacement_table_filepath,
        retirement_curves_table_filepath,
    )

    # We merge everything into a single dictionary
    return faa_forecast_data, us_population_data, airbus_demand_data, fleet_data
