from om_idea.utilities.jax_component import JaxComponent


class AggregateDeltaRpm(JaxComponent):
    def initialize(self):
        # Options
        self.options.declare("num_forecast_years")

    def setup(self):
        # Retrieve options
        num_forecast_years = self.options["num_forecast_years"]

        # Inputs
        self.add_input("relative_delta_rpm_due_to_gdpcc", shape=(num_forecast_years,))
        self.add_input(
            "relative_delta_rpm_due_to_ticket_price",
            shape=(num_forecast_years,),
        )

        # Outputs
        self.add_output("relative_delta_rpm", shape=(num_forecast_years,))

    def jax_compute(self, inputs):
        return {
            "relative_delta_rpm": (1 + inputs["relative_delta_rpm_due_to_gdpcc"])
            * (1 + inputs["relative_delta_rpm_due_to_ticket_price"])
            - 1
        }
