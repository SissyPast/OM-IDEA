import openmdao.api as om

from om_idea.components.legacy_idea import (
    AggregateDeltaRpm,
    AggregateRegionalGdppcImpacts,
    FleetEvolutionModel,
    FleetLevelQuantities,
    GdpPerCapita,
    RelativeDeltaOperatingCosts,
    RelativeDeltaRpmDueToGdppc,
    RelativeDeltaTicketPrice,
    RpmAsm,
    TicketPriceImpactOnRpm,
)
from om_idea.formulations.legacy_idea.namespace import (
    IdeaCoupling,
    IdeaInputs,
    IdeaOptions,
)
from om_idea.utilities.om_helpers import OMComponent, OMGroup, create_group_from_spec


def set_main_loop_solver(group):
    group.linear_solver = om.ScipyKrylov()

    group.nonlinear_solver = om.NewtonSolver(
        solve_subsystems=True,
        maxiter=100,
        iprint=2,
        err_on_non_converge=True,
        rtol=1e-5,
    )

    group.nonlinear_solver.linesearch = om.BoundsEnforceLS()


def create_idea_model(idea_options: IdeaOptions):
    # Individual groups
    relative_delta_rpm_wrt_gdppc = OMGroup(
        name="gdppc_impact_on_rpm",
        subsystems=[
            OMComponent(
                name="domestic_gdppc",
                component=GdpPerCapita(
                    num_forecast_years=idea_options.num_forecast_years
                ),
                inputs_map={
                    "gdp": IdeaInputs.indexed_domestic_gdp,
                    "population": IdeaInputs.indexed_domestic_population,
                },
                outputs_map={"gdppc": IdeaCoupling.indexed_domestic_gdppc},
            ),
            OMComponent(
                name="gdppc_impact_on_domestic_rpm",
                component=RelativeDeltaRpmDueToGdppc(
                    num_forecast_years=idea_options.num_forecast_years,
                    num_trips_vs_gdppc_slope=idea_options.num_trips_vs_gdppc_slope,
                    initial_num_trips=idea_options.initial_domestic_num_trips_per_capita,
                    initial_gdppc=idea_options.initial_domestic_gdppc,
                    indexed_reference_gdppc=idea_options.indexed_reference_domestic_gdppc,
                ),
                inputs_map={"indexed_gdppc": IdeaCoupling.indexed_domestic_gdppc},
                outputs_map={
                    "relative_delta_rpm": IdeaCoupling.relative_delta_domestic_rpm_gdppc,
                },
            ),
            OMComponent(
                name="gdppc_impact_on_atlantic_rpm",
                component=RelativeDeltaRpmDueToGdppc(
                    num_forecast_years=idea_options.num_forecast_years,
                    num_trips_vs_gdppc_slope=idea_options.num_trips_vs_gdppc_slope,
                    initial_num_trips=idea_options.initial_atlantic_num_trips_per_capita,
                    initial_gdppc=idea_options.initial_atlantic_gdppc,
                    indexed_reference_gdppc=idea_options.indexed_reference_atlantic_gdppc,
                ),
                inputs_map={"indexed_gdppc": IdeaInputs.indexed_atlantic_gdppc},
                outputs_map={
                    "relative_delta_rpm": IdeaCoupling.relative_delta_atlantic_rpm_gdppc,
                },
            ),
            OMComponent(
                name="gdppc_impact_on_pacific_rpm",
                component=RelativeDeltaRpmDueToGdppc(
                    num_forecast_years=idea_options.num_forecast_years,
                    num_trips_vs_gdppc_slope=idea_options.num_trips_vs_gdppc_slope,
                    initial_num_trips=idea_options.initial_pacific_num_trips_per_capita,
                    initial_gdppc=idea_options.initial_pacific_gdppc,
                    indexed_reference_gdppc=idea_options.indexed_reference_pacific_gdppc,
                ),
                inputs_map={"indexed_gdppc": IdeaInputs.indexed_pacific_gdppc},
                outputs_map={
                    "relative_delta_rpm": IdeaCoupling.relative_delta_pacific_rpm_gdppc,
                },
            ),
            OMComponent(
                name="gdppc_impact_on_latam_rpm",
                component=RelativeDeltaRpmDueToGdppc(
                    num_forecast_years=idea_options.num_forecast_years,
                    num_trips_vs_gdppc_slope=idea_options.num_trips_vs_gdppc_slope,
                    initial_num_trips=idea_options.initial_latam_num_trips_per_capita,
                    initial_gdppc=idea_options.initial_latam_gdppc,
                    indexed_reference_gdppc=idea_options.indexed_reference_latam_gdppc,
                ),
                inputs_map={"indexed_gdppc": IdeaInputs.indexed_latam_gdppc},
                outputs_map={
                    "relative_delta_rpm": IdeaCoupling.relative_delta_latam_rpm_gdppc,
                },
            ),
            OMComponent(
                name="gdppc_impact_on_international_rpm",
                component=AggregateRegionalGdppcImpacts(
                    num_forecast_years=idea_options.num_forecast_years,
                    initial_atlantic_rpm=idea_options.initial_atlantic_rpm,
                    initial_pacific_rpm=idea_options.initial_pacific_rpm,
                    initial_latam_rpm=idea_options.initial_latam_rpm,
                    indexed_reference_atlantic_rpm=idea_options.indexed_reference_atlantic_rpm,
                    indexed_reference_pacific_rpm=idea_options.indexed_reference_pacific_rpm,
                    indexed_reference_latam_rpm=idea_options.indexed_reference_latam_rpm,
                ),
                inputs_map={
                    "relative_delta_atlantic_rpm": IdeaCoupling.relative_delta_atlantic_rpm_gdppc,
                    "relative_delta_pacific_rpm": IdeaCoupling.relative_delta_pacific_rpm_gdppc,
                    "relative_delta_latam_rpm": IdeaCoupling.relative_delta_latam_rpm_gdppc,
                },
                outputs_map={
                    "relative_delta_international_rpm": IdeaCoupling.relative_delta_international_rpm_gdppc
                },
            ),
        ],
    )

    partial_equilibrium = OMGroup(
        name="partial_equilibrium",
        subsystems=[
            OMComponent(
                name="relative_delta_operating_cost",
                component=RelativeDeltaOperatingCosts(
                    num_forecast_years=idea_options.num_forecast_years,
                    indexed_reference_fuel_intensity=idea_options.indexed_reference_fleet_level_fuel_intensity,
                    indexed_reference_fuel_price=idea_options.indexed_reference_fuel_price,
                    reference_fuel_costs_fraction_of_operating_costs=idea_options.reference_fuel_costs_fraction_of_operating_costs,
                ),
                inputs_map={
                    "indexed_fuel_intensity": IdeaCoupling.indexed_fleet_level_fuel_intensity,
                    "indexed_fuel_price": IdeaInputs.indexed_fuel_price,
                },
                outputs_map={
                    "relative_delta_operating_cost": IdeaCoupling.relative_delta_operating_cost
                },
            ),
            OMComponent(
                name="relative_delta_domestic_ticket_price",
                component=RelativeDeltaTicketPrice(
                    num_forecast_years=idea_options.num_forecast_years,
                    reference_load_factor=idea_options.reference_domestic_load_factor,
                ),
                inputs_map={
                    "relative_delta_operating_cost": IdeaCoupling.relative_delta_operating_cost,
                    "load_factor": IdeaInputs.domestic_load_factor,
                    "pass_through_fraction": IdeaInputs.pass_through_fraction,
                },
                outputs_map={
                    "relative_delta_ticket_price": IdeaCoupling.relative_delta_domestic_ticket_price
                },
            ),
            OMComponent(
                name="relative_delta_international_ticket_price",
                component=RelativeDeltaTicketPrice(
                    num_forecast_years=idea_options.num_forecast_years,
                    reference_load_factor=idea_options.reference_international_load_factor,
                ),
                inputs_map={
                    "relative_delta_operating_cost": IdeaCoupling.relative_delta_operating_cost,
                    "load_factor": IdeaInputs.international_load_factor,
                    "pass_through_fraction": IdeaInputs.pass_through_fraction,
                },
                outputs_map={
                    "relative_delta_ticket_price": IdeaCoupling.relative_delta_international_ticket_price
                },
            ),
            OMComponent(
                name="ticket_price_impact_on_domestic_rpm",
                component=TicketPriceImpactOnRpm(
                    num_forecast_years=idea_options.num_forecast_years,
                    reference_business_rpm_fraction=idea_options.reference_business_rpm_fraction,
                    reference_shorthaul_rpm_fraction=idea_options.reference_shorthaul_rpm_fraction,
                    price_elasticities=idea_options.domestic_price_elasticities,
                ),
                inputs_map={
                    "relative_delta_ticket_price": IdeaCoupling.relative_delta_domestic_ticket_price
                },
                outputs_map={
                    "relative_delta_rpm": IdeaCoupling.relative_delta_domestic_rpm_fuel_lf
                },
            ),
            OMComponent(
                name="ticket_price_impact_on_international_rpm",
                component=TicketPriceImpactOnRpm(
                    num_forecast_years=idea_options.num_forecast_years,
                    reference_business_rpm_fraction=idea_options.reference_business_rpm_fraction,
                    reference_shorthaul_rpm_fraction=idea_options.reference_shorthaul_rpm_fraction,
                    price_elasticities=idea_options.international_price_elasticities,
                ),
                inputs_map={
                    "relative_delta_ticket_price": IdeaCoupling.relative_delta_international_ticket_price
                },
                outputs_map={
                    "relative_delta_rpm": IdeaCoupling.relative_delta_international_rpm_fuel_lf
                },
            ),
            OMComponent(
                name="relative_delta_domestic_rpm",
                component=AggregateDeltaRpm(
                    num_forecast_years=idea_options.num_forecast_years
                ),
                inputs_map={
                    "relative_delta_rpm_due_to_gdpcc": IdeaCoupling.relative_delta_domestic_rpm_gdppc,
                    "relative_delta_rpm_due_to_ticket_price": IdeaCoupling.relative_delta_domestic_rpm_fuel_lf,
                },
                outputs_map={
                    "relative_delta_rpm": IdeaCoupling.relative_delta_domestic_rpm
                },
            ),
            OMComponent(
                name="relative_delta_international_rpm",
                component=AggregateDeltaRpm(
                    num_forecast_years=idea_options.num_forecast_years
                ),
                inputs_map={
                    "relative_delta_rpm_due_to_gdpcc": IdeaCoupling.relative_delta_international_rpm_gdppc,
                    "relative_delta_rpm_due_to_ticket_price": IdeaCoupling.relative_delta_international_rpm_fuel_lf,
                },
                outputs_map={
                    "relative_delta_rpm": IdeaCoupling.relative_delta_international_rpm
                },
            ),
            OMComponent(
                name="updated_domestic_rpm_asm",
                component=RpmAsm(
                    num_forecast_years=idea_options.num_forecast_years,
                    initial_rpm=idea_options.initial_domestic_rpm,
                    indexed_reference_rpm=idea_options.indexed_reference_domestic_rpm,
                ),
                inputs_map={
                    "relative_delta_rpm": IdeaCoupling.relative_delta_domestic_rpm,
                    "load_factor": IdeaInputs.domestic_load_factor,
                },
                outputs_map={
                    "rpm": IdeaCoupling.domestic_rpm,
                    "asm": IdeaCoupling.domestic_asm,
                },
            ),
            OMComponent(
                name="updated_international_rpm_asm",
                component=RpmAsm(
                    num_forecast_years=idea_options.num_forecast_years,
                    initial_rpm=idea_options.initial_international_rpm,
                    indexed_reference_rpm=idea_options.indexed_reference_international_rpm,
                ),
                inputs_map={
                    "relative_delta_rpm": IdeaCoupling.relative_delta_international_rpm,
                    "load_factor": IdeaInputs.international_load_factor,
                },
                outputs_map={
                    "rpm": IdeaCoupling.international_rpm,
                    "asm": IdeaCoupling.international_asm,
                },
            ),
            OMComponent(
                name="fleet_evolution",
                component=FleetEvolutionModel(
                    start_year=idea_options.start_year,
                    end_year=idea_options.end_year,
                    indexed_reference_fuel_intensity=idea_options.indexed_reference_fleet_level_fuel_intensity,
                    aircraft_types_table=idea_options.aircraft_types_table,
                    aircraft_asm_shares_table=idea_options.aircraft_asm_shares_table,
                    deliveries_table=idea_options.deliveries_table,
                    replacement_table=idea_options.replacement_table,
                    retirement_curves_table=idea_options.retirement_curves_table,
                ),
                inputs_map={
                    "domestic_asm": IdeaCoupling.domestic_asm,
                    "international_asm": IdeaCoupling.international_asm,
                    "indexed_fuel_intensity_operations": IdeaInputs.indexed_fleet_level_fuel_intensity_operations,
                },
                outputs_map={
                    "age_distribution_by_year": IdeaCoupling.age_distribution_by_year,
                    "indexed_fuel_intensity_technology": IdeaCoupling.indexed_fleet_level_fuel_intensity_technology,
                    "indexed_fuel_intensity": IdeaCoupling.indexed_fleet_level_fuel_intensity,
                    "relative_fuel_intensity": IdeaCoupling.relative_fleet_level_fuel_intensity,
                },
            ),
        ],
        config_function=set_main_loop_solver,
    )

    idea_spec = OMGroup(
        name="idea",
        subsystems=[
            relative_delta_rpm_wrt_gdppc,
            partial_equilibrium,
            OMComponent(
                name="fleet_level_quantities",
                component=FleetLevelQuantities(
                    num_forecast_years=idea_options.num_forecast_years,
                    initial_fuel_intensity=idea_options.initial_fleet_level_fuel_intensity,
                ),
                inputs_map={
                    "domestic_asm": IdeaCoupling.domestic_asm,
                    "international_asm": IdeaCoupling.international_asm,
                    "indexed_fuel_intensity": IdeaCoupling.indexed_fleet_level_fuel_intensity,
                },
                outputs_map={
                    "fuel_quantity": IdeaCoupling.fuel_quantity,
                    "energy_quantity": IdeaCoupling.energy_quantity,
                    "carbon_dioxide_quantity": IdeaCoupling.carbon_dioxide_quantity,
                },
            ),
        ],
    )

    return create_group_from_spec(idea_spec)
