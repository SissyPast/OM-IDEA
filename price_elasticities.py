import warnings
from dataclasses import KW_ONLY, dataclass, field
from pprint import pprint
from typing import Callable

import openmdao.api as om
from openmdao.core.component import Component


def set_options(om_problem, options):
    # Set option values model-wise
    om_problem.model_options["*"] = options


def set_recorder(om_problem, recorder_filename="cases.sql"):
    # Create and attach a recorder to the problem
    recorder = om.SqliteRecorder(recorder_filename)
    om_problem.model.add_recorder(recorder)


def get_mda_variables(om_problem):
    # We first determine "actual" MDA inputs, otherwise list_inputs() also returns
    # coupling variables that are not real MDA-level inputs

    # Identify all components' inputs
    # Here, we just ignore the warning that says we cannot access variable values at
    # this point. This is okay because we are not relying on any values here, just
    # variable names.
    with warnings.catch_warnings(action="ignore"):
        all_components_inputs = set(
            (
                meta["prom_name"]
                for _, meta in om_problem.model.list_inputs(out_stream=None)
            )
        )

    # Identify all components' outputs
    all_components_outputs = set(
        (
            meta["prom_name"]
            for _, meta in om_problem.model.list_outputs(out_stream=None)
        )
    )

    # Actual MDA inputs are components' inputs that are NOT also components' outputs
    # (this removes the coupling variables from the list of inputs)
    mda_input_variables = all_components_inputs - all_components_outputs
    mda_coupling_variables = all_components_inputs.intersection(all_components_outputs)
    mda_output_variables = all_components_outputs - mda_coupling_variables

    return mda_input_variables, mda_coupling_variables, mda_output_variables


def set_input_values(om_problem, inputs):
    # Get the list of actual MDA inputs
    mda_input_variables, _, _ = get_mda_variables(om_problem)

    # Set inputs' values
    for input_name in mda_input_variables:
        if input_name in inputs:
            om_problem.set_val(input_name, inputs[input_name])
        else:
            print(f"Skipping undefined input: {input_name}")



def set_initial_guesses(om_problem, initial_guesses):
    # Initial guesses
    for variable_name, variable_value in initial_guesses.items():
        om_problem.set_val(variable_name, variable_value)


def generate_n2_diagram(om_problem):
    # Generate an N2 diagram if needed
    # path = om_problem.get_outputs_dir() / "n2.html"
    path = "n2.html"
    om.n2(om_problem, outfile=path, show_browser=False)


def get_output_values(om_problem):
    return {
        output_metadata["prom_name"]: output_metadata["val"]
        for _, output_metadata in om_problem.model.list_outputs(out_stream=None)
    }


def run_om_group(
    om_group,
    inputs,
    global_options=None,
    initial_guesses=None,
    name=None,
    record=False,
    generate_n2=False,
    list_variables=False,
    check_config=False,
    run=True,
):
    # Instantiate problem
    prob = om.Problem(om_group, name=name, reports=None)

    # Set options
    if global_options is not None:
        set_options(prob, global_options)

    # Set recorder
    if record:
        set_recorder(prob, "recorder_db.sql")

    # Run setup
    prob.setup()

    # Check problem
    if check_config:
        prob.check_config()

    # Generate N2 diagram
    if generate_n2:
        generate_n2_diagram(prob)

    # List all variables
    if list_variables:
        mda_input_variables, mda_coupling_variables, mda_output_variables = (
            get_mda_variables(prob)
        )
        print("\n\n=== Input Variables ===")
        pprint(mda_input_variables)
        print("\n\n=== Coupling Variables ===")
        pprint(mda_coupling_variables)
        print("\n\n=== Output Variables ===")
        pprint(mda_output_variables)

    if run:
        # Set input values
        set_input_values(prob, inputs)

        # Set initial guesses if provided
        if initial_guesses is not None:
            set_initial_guesses(prob, initial_guesses)

        # Solve the MDA
        prob.run_model()

        # Retrieve and return output values
        return get_output_values(prob)


def run_single_component(om_component, options, inputs, component_name="om_component"):
    model = om.Group()
    model.add_subsystem(component_name, om_component, promotes=["*"])
    return run_om_group(model, options, inputs)


@dataclass
class OMSystem:
    name: str
    _: KW_ONLY
    inputs_map: dict | None = None
    outputs_map: dict | None = None


@dataclass
class OMGroup(OMSystem):
    subsystems: list[OMSystem]
    config_function: Callable[["OMGroup"], None] | None = None


@dataclass
class OMComponent(OMSystem):
    component: Component


def create_group_from_spec(group_spec: OMGroup):
    """Function that process as nested dict describing the OpenMDAO problem and creates
    the corresponding OpenMDAO groups as needed."""

    # Group to be populated
    group = om.Group()

    # Call config function if provided
    if group_spec.config_function is not None:
        group_spec.config_function(group)

    # Iterate over the group's subsystems
    for subsystem_spec in group_spec.subsystems:
        if isinstance(subsystem_spec, OMComponent):
            subsystem = subsystem_spec.component
        elif isinstance(subsystem_spec, OMGroup):
            subsystem = create_group_from_spec(subsystem_spec)
        else:
            raise RuntimeError("Subsysten is not of type `OMComponent` or `OMGroup`.")

        # We promote everything by default
        promotes_inputs = (
            list(subsystem_spec.inputs_map.items())
            if subsystem_spec.inputs_map is not None
            else ["*"]
        )
        promotes_outputs = (
            list(subsystem_spec.outputs_map.items())
            if subsystem_spec.outputs_map is not None
            else ["*"]
        )

        # Add the subsystem
        group.add_subsystem(
            subsystem_spec.name,
            subsystem,
            promotes_inputs=promotes_inputs,
            promotes_outputs=promotes_outputs,
        )

    return group
