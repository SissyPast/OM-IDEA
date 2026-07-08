import jax.numpy as jnp

from om_idea.utilities.jax_component import JaxComponent


class MarketShares(JaxComponent):
    def initialize(self):
        # Options
        self.options.declare("num_forecast_years")
        self.options.declare("num_options")

    def setup(self):
        # Retrieve options
        num_forecast_years = self.options["num_forecast_years"]
        num_options = self.options["num_options"]

        # Inputs
        self.add_input("share_weights", shape=(num_options,))
        self.add_input("cost_sensitivity", shape=(1,))
        self.add_input("prices", shape=(num_forecast_years, num_options))

        # Outputs
        self.add_output("market_shares", shape=(num_forecast_years, num_options))

    def jax_compute(self, inputs):
        cost_sensitivity = inputs["cost_sensitivity"].reshape(-1, 1)
        numerator = inputs["share_weights"] * jnp.exp(
            -cost_sensitivity * inputs["prices"]
        )
        denominator = jnp.sum(numerator, axis=1, keepdims=True)

        return {"market_shares": numerator / denominator}
