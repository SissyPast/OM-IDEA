# import jax.numpy as jnp

# from om_idea.utilities.jax_component import JaxComponent


# class EnergyPrice(JaxComponent):
#     def initialize(self):
#         self.options.declare("num_years")
#         self.options.declare("num_energy_carriers")
#         self.options.declare("num_types")
#         self.options.declare("num_missions")
#         self.options.declare("num_age_bins")

#     def setup(self):
#         # Options
#         num_years = self.options["num_years"]
#         num_energy_carriers = self.options["num_energy_carriers"]
#         num_types = self.options["num_types"]
#         num_missions = self.options["num_missions"]
#         num_age_bins = self.options["num_age_bins"]

#         # Inputs
#         self.add_input(
#             "asm_fractions_per_year",
#             shape=(num_years, num_types, num_missions, num_age_bins),
#         )
#         self.add_input(
#             "energy_carrier_breakdown_per_type",
#             shape=(num_types, num_energy_carriers),
#         )
#         self.add_input(
#             "energy_price_per_energy_source", shape=(num_years, num_energy_carriers)
#         )

#         # Outputs
#         self.add_output(
#             "yearly_fleet_energy_use_breakdown", shape=(num_years, num_energy_carriers)
#         )
#         self.add_output("yearly_average_energy_price", shape=(num_years,))
#         self.add_output("indexed_energy_price", shape=(num_years,))

#     def jax_compute(self, inputs):
#         energy_use_breakdown = jnp.einsum(
#             "ijkl,jm->im",
#             inputs["asm_fractions_per_year"],
#             inputs["energy_carrier_breakdown_per_type"],
#         )

#         average_energy_price = jnp.einsum(
#             "ij,ij->i",
#             energy_use_breakdown,
#             inputs["energy_price_per_energy_source"],
#         )

#         indexed_energy_price = average_energy_price / average_energy_price[0]

#         return {
#             "yearly_fleet_energy_use_breakdown": energy_use_breakdown,
#             "yearly_average_energy_price": average_energy_price,
#             "indexed_energy_price": indexed_energy_price,
#         }
