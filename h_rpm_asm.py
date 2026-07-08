import jax.numpy as jnp

from om_idea.utilities.jax_component import JaxComponent


def relative_demand(relative_ticket_price, price_elasticity):
    frac = (relative_ticket_price - 1) / (relative_ticket_price + 1)
    return (1 + price_elasticity * frac) / (1 - price_elasticity * frac)


class TicketPriceImpactOnRpm(JaxComponent):
    def initialize(self):
        # Options
        self.options.declare("num_forecast_years")
        self.options.declare("reference_business_rpm_fraction")
        self.options.declare("reference_shorthaul_rpm_fraction")
        self.options.declare("price_elasticities")

    def setup(self):
        # Retrieve options
        num_forecast_years = self.options["num_forecast_years"]

        # Inputs
        self.add_input("relative_delta_ticket_price", shape=(num_forecast_years,))

        # Outputs
        self.add_output("relative_delta_rpm", shape=(num_forecast_years,))

    def jax_compute(self, inputs):
        # All fractions
        purpose_fractions = {
            "business": self.options["reference_business_rpm_fraction"],
            "leisure": 1 - self.options["reference_business_rpm_fraction"],
        }
        distance_fractions = {
            "shorthaul": self.options["reference_shorthaul_rpm_fraction"],
            "longhaul": 1 - self.options["reference_shorthaul_rpm_fraction"],
        }

        relative_delta_rpm = jnp.sum(
            jnp.stack(
                [
                    purpose_fraction
                    * distance_fraction
                    * (
                        relative_demand(
                            1 + inputs["relative_delta_ticket_price"],
                            self.options["price_elasticities"][purpose][distance],
                        )
                        - 1
                    )
                    for purpose, purpose_fraction in purpose_fractions.items()
                    for distance, distance_fraction in distance_fractions.items()
                ],
                axis=1,
            ),
            axis=1,
        )

        return {"relative_delta_rpm": relative_delta_rpm}
