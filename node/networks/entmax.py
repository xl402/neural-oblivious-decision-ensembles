import tensorflow as tf


@tf.function
def sparsemoid(inputs: tf.Tensor):
    return tf.clip_by_value(0.5 * inputs + 0.5, 0., 1.)


def entmax15(inputs, axis=-1):
    """
    Entmax 1.5 implementation
    paper: https://arxiv.org/pdf/1905.05702.pdf
    """
    @tf.custom_gradient
    def _entmax_inner(inputs):
        with tf.name_scope('entmax'):
            inputs = inputs / 2
            inputs -= tf.reduce_max(inputs, axis, keepdims=True)

            threshold, _ = entmax_threshold_and_support(inputs, axis)
            outputs_sqrt = tf.nn.relu(inputs - threshold)
            outputs = tf.square(outputs_sqrt)

        def grad_fn(d_outputs):
            with tf.name_scope('entmax_grad'):
                d_inputs = d_outputs * outputs_sqrt
                q = tf.reduce_sum(d_inputs, axis=axis, keepdims=True)
                q = q / tf.reduce_sum(outputs_sqrt, axis=axis, keepdims=True)
                d_inputs -= q * outputs_sqrt
                return d_inputs
        return outputs, grad_fn
    return _entmax_inner(inputs)


def entmax_threshold_and_support(inputs, axis=-1):
    """
    Computes clipping threshold for entmax1.5 over specified axis
    """
    num_outcomes = tf.shape(inputs)[axis]
    inputs_sorted, _ = _top_k_over_axis(inputs, k=num_outcomes, axis=axis)

    rho = _make_ix_like(inputs, axis=axis)

    mean = tf.cumsum(inputs_sorted, axis=axis) / rho

    mean_sq = tf.cumsum(tf.square(inputs_sorted), axis=axis) / rho
    delta = (1 - rho * (mean_sq - tf.square(mean))) / rho

    delta_nz = tf.nn.relu(delta)
    tau = mean - tf.sqrt(delta_nz)

    comparator = tf.cast(tau <= inputs_sorted, dtype=tf.int64)
    support_size = tf.reduce_sum(comparator, axis=axis, keepdims=True)
    tau_star = _gather_over_axis(tau, support_size - 1, axis)
    # tau_star = tf.gather(tau, [inputs.shape[axis]-1], axis=axis)
    return tau_star, support_size


def _top_k_over_axis(inputs, k, axis=-1):
    """ performs tf.nn.top_k over any chosen axis """
    if axis == -1:
        return tf.nn.top_k(inputs, k, sorted=True)

    perm_order = list(range(inputs.shape.ndims))
    perm_order.append(perm_order.pop(axis))
    inv_order = [perm_order.index(i) for i in range(len(perm_order))]

    input_perm = tf.transpose(inputs, perm_order)
    input_perm_sorted, sort_indices_perm = tf.nn.top_k(input_perm, k=k, sorted=True)

    input_sorted = tf.transpose(input_perm_sorted, inv_order)
    sort_indices = tf.transpose(sort_indices_perm, inv_order)
    return input_sorted, sort_indices


def _make_ix_like(inputs, axis=-1):
    """ creates indices 0, ... , input[axis] unsqueezed to input dimensios """
    assert inputs.shape.ndims is not None
    rho = tf.cast(tf.range(1, tf.shape(inputs)[axis] + 1), dtype=inputs.dtype)
    view = [1] * inputs.shape.ndims
    view[axis] = -1
    return tf.reshape(rho, view)


def _gather_over_axis(values, indices, gather_axis):
    assert indices.shape.ndims is not None
    assert indices.shape.ndims == values.shape.ndims

    ndims = indices.shape.ndims
    gather_axis = gather_axis % ndims
    shape = tf.shape(indices)

    selectors = []
    for axis_i in range(ndims):
        if axis_i == gather_axis:
            selectors.append(indices)
        else:
            index_i = tf.range(tf.cast(shape[axis_i], dtype=indices.dtype), dtype=indices.dtype)
            index_i = tf.reshape(index_i, [-1 if i == axis_i else 1 for i in range(ndims)])
            index_i = tf.tile(index_i, [shape[i] if i != axis_i else 1 for i in range(ndims)])
            selectors.append(index_i)

    return tf.gather_nd(values, tf.stack(selectors, axis=-1))
