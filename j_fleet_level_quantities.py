from om_idea.utilities.jax_component import JaxComponent


class RpmAsm(JaxComponent):
    def initialize(self):
        # Options
        self.options.declare("num_forecast_years")
        self.options.declare("initial_rpm")
        self.options.declare("indexed_reference_rpm")

    def setup(self):
        # Retrieve options
        num_forecast_years = self.options["num_forecast_years"]

        # Inputs
        self.add_input("relative_delta_rpm", shape=(num_forecast_years,))
        self.add_input("load_factor", shape=(num_forecast_years,))

        # Outputs
        self.add_output("rpm", shape=(num_forecast_years,))
        self.add_output("asm", shape=(num_forecast_years,))

    def jax_compute(self, inputs):
        rpm = (
            self.options["initial_rpm"]
            * self.options["indexed_reference_rpm"]
            * (1 + inputs["relative_delta_rpm"])
        )

        return {
            "rpm": rpm,
            "asm": rpm / inputs["load_factor"],
        }
