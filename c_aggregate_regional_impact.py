from om_idea.utilities.jax_component import JaxComponent


class GdpPerCapita(JaxComponent):
    def initialize(self):
        # Options
        self.options.declare("num_forecast_years")

    def setup(self):
        # Retrieve options
        num_forecast_years = self.options["num_forecast_years"]

        # Inputs
        self.add_input("gdp", shape=(num_forecast_years,))
        self.add_input("population", shape=(num_forecast_years,))

        # Outputs
        self.add_output("gdppc", shape=(num_forecast_years,))

    def jax_compute(self, inputs):
        return {"gdppc": inputs["gdp"] / inputs["population"]}
