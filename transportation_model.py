def prepare_mda_options(
    num_forecast_years, faa_forecast, us_population_forecast, airbus_data
):
    baseline_projections = {
        # Domestic aviation data
        "baseline_domestic_rpm": faa_forecast["domestic_rpm"],
        "baseline_domestic_asm": faa_forecast["domestic_asm"],
        "baseline_domestic_load_factor": faa_forecast["domestic_load_factor"],
        "baseline_domestic_fuel_quantity": faa_forecast["domestic_fuel_quantity"],
        # International aviation data
        "baseline_international_rpm": faa_forecast["international_rpm"],
        "baseline_international_asm": faa_forecast["international_asm"],
        "baseline_international_load_factor": faa_forecast["international_load_factor"],
        "baseline_international_fuel_quantity": faa_forecast[
            "international_fuel_quantity"
        ],
        # Economic data
        "baseline_domestic_gdp": faa_forecast["domestic_gdp"],
        "baseline_regional_gdp_pacific": faa_forecast["regional_gdp_pacific"],
        "baseline_regional_gdp_atlantic": faa_forecast["regional_gdp_atlantic"],
        "baseline_regional_gdp_latam": faa_forecast["regional_gdp_latam"],
        # Regional projections
        "baseline_regional_rpm_pacific": faa_forecast["regional_rpm_pacific"],
        "baseline_regional_rpm_atlantic": faa_forecast["regional_rpm_atlantic"],
        "baseline_regional_rpm_latam": faa_forecast["regional_rpm_latam"],
        # US population
        "baseline_domestic_population": us_population_forecast["us_population_count"],
    }

    # We only keep the forecast values, not the historical ones
    for key, value in baseline_projections.items():
        baseline_projections[key] = value[-num_forecast_years:]

    # Airbus data
    airbus_data = {
        "airbus_trips_vs_gdp_data": airbus_data["trips_vs_gdp_data"],
        "airbus_regional_data": airbus_data["regional_data"],
    }

    return {**baseline_projections, **airbus_data}
