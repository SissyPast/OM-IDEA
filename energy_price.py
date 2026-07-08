from om_idea.utilities.jax_component import JaxComponent

LITERS_PER_GALLON = 3.78541178  # l/gal
CONVENTIONAL_JET_FUEL_DENSITY = 0.802  # kg/l
CONVENTIONAL_JET_FUEL_HEATING_VALUE = 43.20  # MJ/kg
CONVENTIONAL_JET_FUEL_CO2_INTENSITY = 87.5 / 1000  # kg/MJ


class FleetLevelQuantities(JaxComponent):
    def initialize(self):
        self.options.declare("num_forecast_years")
        self.options.declare("initial_fuel_intensity")

    def setup(self):
        # Retrieve options
        num_forecast_years = self.options["num_forecast_years"]

        # Inputs
        self.add_input("domestic_asm", shape=(num_forecast_years,))
        self.add_input("international_asm", shape=(num_forecast_years,))
        self.add_input("indexed_fuel_intensity", shape=(num_forecast_years,))

        # Outputs
        self.add_output("fuel_quantity", shape=(num_forecast_years))
        self.add_output("energy_quantity", shape=(num_forecast_years))
        self.add_output("carbon_dioxide_quantity", shape=(num_forecast_years))

    def jax_compute(self, inputs):
        fuel_quantity = (
            (inputs["domestic_asm"] + inputs["international_asm"])
            * inputs["indexed_fuel_intensity"]
            * self.options["initial_fuel_intensity"]
        )

        energy_quantity = (
            fuel_quantity
            * LITERS_PER_GALLON
            * CONVENTIONAL_JET_FUEL_DENSITY
            * CONVENTIONAL_JET_FUEL_HEATING_VALUE
        )

        carbon_dioxide_quantity = energy_quantity * CONVENTIONAL_JET_FUEL_CO2_INTENSITY

        return {
            "fuel_quantity": fuel_quantity,
            "energy_quantity": energy_quantity,
            "carbon_dioxide_quantity": carbon_dioxide_quantity,
        }
