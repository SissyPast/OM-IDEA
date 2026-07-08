import jax.numpy as jnp

from om_idea.utilities.jax_component import JaxComponent


class ShareWeights(JaxComponent):
    def initialize(self):
        # Options
        self.options.declare("num_forecast_years")
        self.options.declare("num_options")

    def setup(self):
        # Retrieve options
        num_forecast_years = self.options["num_forecast_years"]
        num_options = self.options["num_options"]

        # Inputs
        self.add_input("initial_market_shares", shape=(num_options,))
        self.add_input("cost_sensitivity", shape=(1,))
        self.add_input("prices", shape=(num_forecast_years, num_options))

        # Outputs
        self.add_output("share_weights", shape=(num_options,))

    def jax_compute(self, inputs):
        relative_alphas = (
            inputs["initial_market_shares"] / inputs["initial_market_shares"][0]
        ) * jnp.exp(
            inputs["cost_sensitivity"]
            * (inputs["prices"][0, :] - inputs["prices"][0, 0])
        )
        return {"share_weights": relative_alphas / jnp.sum(relative_alphas)}
