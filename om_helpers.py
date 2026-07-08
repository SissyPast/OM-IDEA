import warnings
from itertools import product

import numpy as np
import openmdao.api as om
from jax import jacfwd, jacrev, jit, tree_util


class JaxComponent(om.ExplicitComponent):
    def jax_compute(self, inputs):
        return NotImplementedError

    def jitted_compute(self, inputs):
        return jit(self.jax_compute)(inputs)

    # def setup_partials(self):
    #     # Retrieve inputs and outputs metadata
    #     # Here, we just ignore the warning that says we cannot access variable values at
    #     # this point. This is okay because we are not relying on any values here, just
    #     # variable names.
    #     with warnings.catch_warnings(action="ignore"):
    #         inputs_metadata = self.list_inputs(
    #             shape=True, out_stream=None, return_format="dict"
    #         )
    #     outputs_metadata = self.list_outputs(
    #         shape=True, out_stream=None, return_format="dict"
    #     )

    #     # Compute inputs and outputs dimension
    #     inputs_dimension = np.sum(
    #         [np.prod(input_["shape"]) for input_ in inputs_metadata.values()]
    #     )
    #     outputs_dimension = np.sum(
    #         [np.prod(output["shape"]) for output in outputs_metadata.values()]
    #     )

    #     # Jacobian depends on dimension
    #     if inputs_dimension < outputs_dimension:
    #         self.jacobian = jacfwd(self.jitted_compute)
    #     else:
    #         self.jacobian = jacrev(self.jitted_compute)

    #     # Keep track of input and output names
    #     self.input_names = inputs_metadata.keys()
    #     self.output_names = outputs_metadata.keys()

    #     # All partial derivatives will be computed
    #     self.declare_partials("*", "*")

    def compute(self, inputs, outputs):
        # Run the jitted function
        _outputs = self.jitted_compute(dict(inputs.items()))

        # Populate the outputs
        for output_name, output_value in _outputs.items():
            outputs[output_name] = output_value

    def compute_partials(self, inputs, partials):
        # Run the AD jacobian
        _partials = self.jacobian(dict(inputs.items()))

        # Populate the Jacobian matrices
        for output_name, input_name in product(self.output_names, self.input_names):
            if len(_partials[output_name][input_name].shape) == 1:
                _partials[output_name][input_name] = np.diag(
                    _partials[output_name][input_name]
                )
            partials[output_name, input_name] = _partials[output_name][input_name]

    def _tree_flatten(self):
        children = tuple()  # inputs are the only dynamic values
        aux_data = dict(self.options)  # options are the only static
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        jax_component = cls(*children, **aux_data)
        jax_component.setup()
        jax_component.setup_partials()
        return jax_component


# Register the JaxedComponent as a pytree
tree_util.register_pytree_node(
    JaxComponent, JaxComponent._tree_flatten, JaxComponent._tree_unflatten
)
