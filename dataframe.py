def check_inputs_shapes(inputs_shapes, input_values):
    for input_name, input_shape in inputs_shapes.items():
        # Check input is present
        assert input_name in input_values, f'Input "{input_name}" is missing.'

        # Check input shape
        given_value_shape = input_values[input_name].shape
        assert given_value_shape == input_shape, (
            f'Input "{input_name}" has the wrong shape: given value has shape {given_value_shape} but it should be {input_shape}.'
        )
