from om_idea.utilities.jax_component import JaxComponent


class RelativeDeltaRpmDueToGdppc(JaxComponent):
    def initialize(self):
        # Options
        self.options.declare("num_forecast_years")
        self.options.declare("num_trips_vs_gdppc_slope")
        self.options.declare("initial_num_trips")
        self.options.declare("initial_gdppc")
        self.options.declare("indexed_reference_gdppc")

    def setup(self):
        # Retrieve options
        num_forecast_years = self.options["num_forecast_years"]

        # Inputs
        self.add_input("indexed_gdppc", shape=(num_forecast_years,))

        # Outputs
        self.add_output("relative_delta_rpm", shape=(num_forecast_years,))

    def jax_compute(self, inputs):
        n_0 = self.options["initial_num_trips"]
        g_0 = self.options["initial_gdppc"]
        b = self.options["num_trips_vs_gdppc_slope"]
        g_ref_indexed = self.options["indexed_reference_gdppc"]
        factor = g_ref_indexed / (n_0 / (b * g_0) + g_ref_indexed - 1)

        return {
            "relative_delta_rpm": factor * (inputs["indexed_gdppc"] / g_ref_indexed - 1)
        }
