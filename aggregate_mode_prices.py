import jax.numpy as jnp

from om_idea.utilities.jax_component import JaxComponent
from om_idea.utilities.timeseries import compute_yearly_growth_rate


class FleetGrowth(JaxComponent):
    def initialize(self):
        self.options.declare("num_years")
        self.options.declare("num_missions")
        self.options.declare("num_domestic_missions")
        self.options.declare("num_international_missions")

    def setup(self):
        # Options
        num_years = self.options["num_years"]
        num_domestic_missions = self.options["num_domestic_missions"]
        num_international_missions = self.options["num_international_missions"]
        num_missions = num_domestic_missions + num_international_missions

        # Inputs
        self.add_input("domestic_asm", shape=(num_years,))
        self.add_input(
            "domestic_missions_fractions",
            shape=(
                num_years,
                num_domestic_missions,
            ),
        )
        self.add_input("international_asm", shape=(num_years,))
        self.add_input(
            "international_missions_fractions",
            shape=(
                num_years,
                num_international_missions,
            ),
        )

        # Outputs
        self.add_output("yearly_fleet_growth", shape=(num_years - 1, num_missions))

    def jax_compute(self, inputs):
        domestic_growth = compute_yearly_growth_rate(inputs["domestic_asm"])
        international_growth = compute_yearly_growth_rate(inputs["international_asm"])

        print(domestic_growth.shape)
        print(inputs["domestic_missions_fractions"].shape)
        print(international_growth.shape)
        print(inputs["international_missions_fractions"].shape)

        return {
            "yearly_fleet_growth": jnp.concatenate(
                (
                    domestic_growth[:, jnp.newaxis]
                    * jnp.ones((1, self.options["num_domestic_missions"])),
                    international_growth[:, jnp.newaxis]
                    * jnp.ones((1, self.options["num_international_missions"])),
                ),
                axis=1,
            )
        }

        # return {
        #     "yearly_fleet_growth": jnp.concatenate(
        #         (
        #             domestic_growth[:, jnp.newaxis]
        #             * inputs["domestic_missions_fractions"][:-1, :],
        #             international_growth[:, jnp.newaxis]
        #             * inputs["international_missions_fractions"][:-1, :],
        #         ),
        #         axis=1,
        #     )
        # }
