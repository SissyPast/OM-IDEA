from om_idea.utilities.jax_component import JaxComponent


class AggregateRegionalGdppcImpacts(JaxComponent):
    def initialize(self):
        # Options
        self.options.declare("num_forecast_years")

        self.options.declare("initial_atlantic_rpm")
        self.options.declare("initial_pacific_rpm")
        self.options.declare("initial_latam_rpm")
        self.options.declare("indexed_reference_atlantic_rpm")
        self.options.declare("indexed_reference_pacific_rpm")
        self.options.declare("indexed_reference_latam_rpm")

    def setup(self):
        # Retrieve options
        num_forecast_years = self.options["num_forecast_years"]

        # Inputs
        self.add_input("relative_delta_atlantic_rpm", shape=(num_forecast_years,))
        self.add_input("relative_delta_pacific_rpm", shape=(num_forecast_years,))
        self.add_input("relative_delta_latam_rpm", shape=(num_forecast_years,))

        # Outputs
        self.add_output("relative_delta_international_rpm", shape=(num_forecast_years,))

    def jax_compute(self, inputs):
        reference_atlantic_rpm = (
            self.options["initial_atlantic_rpm"]
            * self.options["indexed_reference_atlantic_rpm"]
        )
        reference_pacific_rpm = (
            self.options["initial_pacific_rpm"]
            * self.options["indexed_reference_pacific_rpm"]
        )
        reference_latam_rpm = (
            self.options["initial_latam_rpm"]
            * self.options["indexed_reference_latam_rpm"]
        )

        return {
            "relative_delta_international_rpm": (
                reference_atlantic_rpm * inputs["relative_delta_atlantic_rpm"]
                + reference_pacific_rpm * inputs["relative_delta_pacific_rpm"]
                + reference_latam_rpm * inputs["relative_delta_latam_rpm"]
            )
            / (reference_atlantic_rpm + reference_pacific_rpm + reference_latam_rpm),
        }
