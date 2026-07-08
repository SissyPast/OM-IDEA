from om_idea.utilities.jax_component import JaxComponent


class RelativeDeltaOperatingCosts(JaxComponent):
    def initialize(self):
        # Options
        self.options.declare("num_forecast_years")
        self.options.declare("indexed_reference_energy_intensity")
        self.options.declare("indexed_reference_energy_price")
        self.options.declare("reference_energy_costs_fraction_of_operating_costs")

    def setup(self):
        # Retrieve options
        num_forecast_years = self.options["num_forecast_years"]

        # Inputs
        self.add_input("indexed_energy_intensity", shape=(num_forecast_years,))
        self.add_input("indexed_energy_price", shape=(num_forecast_years,))

        # Outputs
        self.add_output("relative_delta_operating_cost", shape=(num_forecast_years,))

    def jax_compute(self, inputs):
        relative_fuel_intensity = (
            inputs["indexed_energy_intensity"]
            / self.options["indexed_reference_energy_intensity"]
        )
        relative_fuel_price = (
            inputs["indexed_energy_price"]
            / self.options["indexed_reference_energy_price"]
        )

        return {
            "relative_delta_operating_cost": self.options[
                "reference_energy_costs_fraction_of_operating_costs"
            ]
            * (relative_fuel_intensity * relative_fuel_price - 1)
        }
