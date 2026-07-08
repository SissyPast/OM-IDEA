import jax.numpy as jnp

from om_idea.utilities.jax_component import JaxComponent


class AggregateModePrices(JaxComponent):
    def initialize(self):
        # Options
        self.options.declare("num_forecast_years")

    def setup(self):
        # Retrieve options
        num_forecast_years = self.options["num_forecast_years"]

        # Inputs
        self.add_input("air_prices", shape=(num_forecast_years))
        self.add_input("rail_prices", shape=(num_forecast_years))
        self.add_input("road_prices", shape=(num_forecast_years))

        # Outputs
        self.add_output("mode_prices", shape=(num_forecast_years,3))

    def jax_compute(self, inputs):
        return {
            "mode_prices": jnp.stack(
                [inputs["air_prices"], inputs["rail_prices"], inputs["road_prices"]],
                axis=1,
            )
        }
