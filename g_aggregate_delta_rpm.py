from om_idea.utilities.jax_component import JaxComponent


class RelativeDeltaTicketPrice(JaxComponent):
    def initialize(self):
        # Options
        self.options.declare("num_forecast_years")
        self.options.declare("reference_load_factor")

    def setup(self):
        # Retrieve options
        num_forecast_years = self.options["num_forecast_years"]

        # Inputs
        self.add_input("relative_delta_operating_cost", shape=(num_forecast_years,))
        self.add_input("load_factor", shape=(num_forecast_years,))
        # pass-through is  cost_per_rpm_fraction_of_yield
        self.add_input("pass_through_fraction", shape=(num_forecast_years,))

        # Outputs
        self.add_output("relative_delta_ticket_price", shape=(num_forecast_years,))

    def jax_compute(self, inputs):
        relative_load_factor = (
            inputs["load_factor"] / self.options["reference_load_factor"]
        )
        relative_operating_cost = 1 + inputs["relative_delta_operating_cost"]

        return {
            "relative_delta_ticket_price": (
                relative_operating_cost / relative_load_factor - 1
            )
            * inputs["pass_through_fraction"]
        }
