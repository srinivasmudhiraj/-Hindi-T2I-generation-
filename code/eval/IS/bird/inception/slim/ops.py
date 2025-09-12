# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains convenience wrappers for typical Neural Network TensorFlow layers.

   Additionally it maintains a collection with update_ops that need to be
   updated after the ops have been computed, for exmaple to update moving means
   and moving variances of batch_norm.

   Ops that have different behavior during training or eval have an is_training
   parameter. Additionally Ops that contain variables.variable have a trainable
   parameter, which control if the ops variables are trainable or not.
"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from tensorflow.python.training import moving_averages

from eval.IS.bird.inception.slim import losses
from eval.IS.bird.inception.slim import scopes
from eval.IS.bird.inception.slim import variables
# Used to keep the update ops done by batch_norm.
#UPDATE_OPS_COLLECTION = '_update_ops_'




UPDATE_OPS_COLLECTION = tf.compat.v1.GraphKeys.UPDATE_OPS


@scopes.add_arg_scope
def batch_norm(inputs,
               decay=0.999,
               center=True,
               scale=True,
               epsilon=0.001,
               moving_vars='moving_vars',
               activation=None,
               is_training=True,
               trainable=True,
               restore=True,
               scope=None,
               reuse=None):
    """Adds a Batch Normalization layer.

    Args:
        inputs: Input tensor of size [batch_size, height, width, channels].
        decay: Decay for moving averages.
        center: Whether to subtract beta.
        scale: Whether to multiply by gamma.
        epsilon: Small float added to variance for numerical stability.
        moving_vars: Collection name to use for moving averages.
        activation: Optional activation function.
        is_training: Whether the model is in training mode.
        trainable: Whether variables are trainable.
        restore: Whether the variables should be restored during checkpoint loading.
        scope: Name scope for variable creation.
        reuse: Whether or not to reuse variables.

    Returns:
        A tensor normalized with batch normalization.
    """

    # Get the shape of input tensor
    inputs_shape = inputs.get_shape()
    
    with tf.compat.v1.variable_scope(scope, 'BatchNorm', [inputs], reuse=reuse):

        # Define axis for normalization (last dimension)
        axis = list(range(len(inputs_shape) - 1))

        # Define scale and beta variables
        beta = tf.compat.v1.get_variable(
            'beta',
            shape=inputs_shape[-1:],
            initializer=tf.zeros_initializer(),
            trainable=trainable,
            collections=[moving_vars]
        )

        gamma = tf.compat.v1.get_variable(
            'gamma',
            shape=inputs_shape[-1:],
            initializer=tf.ones_initializer(),
            trainable=trainable,
            collections=[moving_vars]
        )

        # Define moving averages (for inference mode)
        moving_mean = tf.compat.v1.get_variable(
            'moving_mean',
            shape=inputs_shape[-1:],
            initializer=tf.zeros_initializer(),
            trainable=False,
            collections=[moving_vars]
        )
        moving_variance = tf.compat.v1.get_variable(
            'moving_variance',
            shape=inputs_shape[-1:],
            initializer=tf.ones_initializer(),
            trainable=False,
            collections=[moving_vars]
        )

        # Handle training vs inference mode logic
        if is_training:
            # Calculate batch statistics
            mean, variance = tf.nn.moments(inputs, axes=axis)

            # Update moving averages with exponential moving average
            with tf.control_dependencies([
                tf.compat.v1.assign_moving_average(moving_mean, mean, decay),
                tf.compat.v1.assign_moving_average(moving_variance, variance, decay)
            ]):
                outputs = tf.nn.batch_normalization(
                    inputs, mean, variance, beta, gamma, epsilon
                )
        else:
            # Use the moving statistics in inference mode
            outputs = tf.nn.batch_normalization(
                inputs, moving_mean, moving_variance, beta, gamma, epsilon
            )

        # Apply activation function if provided
        if activation:
            outputs = activation(outputs)

        return outputs
    
def _two_element_tuple(int_or_tuple):
  """Converts `int_or_tuple` to height, width.

  Several of the functions that follow accept arguments as either
  a tuple of 2 integers or a single integer.  A single integer
  indicates that the 2 values of the tuple are the same.

  This functions normalizes the input value by always returning a tuple.

  Args:
    int_or_tuple: A list of 2 ints, a single int or a tf.TensorShape.

  Returns:
    A tuple with 2 values.

  Raises:
    ValueError: If `int_or_tuple` it not well formed.
  """
  if isinstance(int_or_tuple, (list, tuple)):
    if len(int_or_tuple) != 2:
      raise ValueError('Must be a list with 2 elements: %s' % int_or_tuple)
    return int(int_or_tuple[0]), int(int_or_tuple[1])
  if isinstance(int_or_tuple, int):
    return int(int_or_tuple), int(int_or_tuple)
  if isinstance(int_or_tuple, tf.TensorShape):
    if len(int_or_tuple) == 2:
      return int_or_tuple[0], int_or_tuple[1]
  raise ValueError('Must be an int, a list with 2 elements or a TensorShape of '
                   'length 2')
@scopes.add_arg_scope
def conv2d(inputs,
           num_filters_out,
           kernel_size,
           stride=1,
           padding='SAME',
           activation_fn=tf.nn.relu,
           stddev=0.01,
           bias=0.0,
           weight_decay=0,
           batch_norm_params=None,
           is_training=True,
           trainable=True,
           restore=True,
           scope=None,
           reuse=None):
    """Applies a 2D convolution with optional batch normalization."""
    
    with tf.compat.v1.variable_scope(scope, 'Conv', [inputs], reuse=reuse):
        # Handle kernel size and stride safely
        kernel_h, kernel_w = _two_element_tuple(kernel_size)
        stride_h, stride_w = _two_element_tuple(stride)
        
        # Determine input channels
        num_filters_in = inputs.get_shape()[-1]
        
        # Define weight shape
        weights_shape = [kernel_h, kernel_w, num_filters_in, num_filters_out]
        
        # Initialize weights with truncated Gaussian
        weights_initializer = tf.compat.v1.truncated_normal_initializer(stddev=stddev)
        
        # Handle L2 regularization if weight_decay is specified
        l2_regularizer = None
        if weight_decay and weight_decay > 0:
            l2_regularizer = losses.get_regularizer_fn(l2_weight=weight_decay)

        # Create weights variable
        weights = variables.variable('weights',
                                     shape=weights_shape,
                                     initializer=weights_initializer,
                                     regularizer=l2_regularizer,
                                     trainable=trainable,
                                     restore=restore)

        # Perform convolution directly without using `ksize`
        conv = tf.nn.conv2d(
            inputs,
            weights,
            strides=[1, int(stride_h), int(stride_w), 1],  # Adjust the strides
            padding=padding
        )
        
        # Handle optional batch normalization
        if batch_norm_params:
            with scopes.arg_scope([batch_norm], is_training=is_training,
                                  trainable=trainable, restore=restore):
                outputs = batch_norm(conv, **batch_norm_params)
        else:
            # If no batch normalization, handle bias directly
            bias_shape = [num_filters_out]
            bias_initializer = tf.constant_initializer(bias)
            biases = variables.variable('biases',
                                        shape=bias_shape,
                                        initializer=bias_initializer,
                                        trainable=trainable,
                                        restore=restore)

            # Apply bias
            outputs = tf.nn.bias_add(conv, biases)

        # Apply activation function if specified
        if activation_fn:
            outputs = activation_fn(outputs)

    return outputs

@scopes.add_arg_scope
def fc(inputs,
       num_units_out,
       activation=tf.nn.relu,
       stddev=0.01,
       bias=0.0,
       weight_decay=0,
       batch_norm_params=None,
       is_training=True,
       trainable=True,
       restore=True,
       scope=None,
       reuse=None):
  """Adds a fully connected layer followed by an optional batch_norm layer.

  FC creates a variable called 'weights', representing the fully connected
  weight matrix, that is multiplied by the input. If `batch_norm` is None, a
  second variable called 'biases' is added to the result of the initial
  vector-matrix multiplication.

  Args:
    inputs: a [B x N] tensor where B is the batch size and N is the number of
            input units in the layer.
    num_units_out: the number of output units in the layer.
    activation: activation function.
    stddev: the standard deviation for the weights.
    bias: the initial value of the biases.
    weight_decay: the weight decay.
    batch_norm_params: parameters for the batch_norm. If is None don't use it.
    is_training: whether or not the model is in training mode.
    trainable: whether or not the variables should be trainable or not.
    restore: whether or not the variables should be marked for restore.
    scope: Optional scope for variable_scope.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.

  Returns:
     the tensor variable representing the result of the series of operations.
  """
  with tf.compat.v1.variable_scope(scope, 'FC', [inputs], reuse=reuse):
    num_units_in = inputs.get_shape()[1]
    weights_shape = [num_units_in, num_units_out]
    weights_initializer = tf.keras.initializers.TruncatedNormal(stddev=stddev)
    l2_regularizer = None
    if weight_decay and weight_decay > 0:
      l2_regularizer = losses.l2_regularizer(weight_decay)
    weights = variables.variable('weights',
                                 shape=weights_shape,
                                 initializer=weights_initializer,
                                 regularizer=l2_regularizer,
                                 trainable=trainable,
                                 restore=restore)
    if batch_norm_params is not None:
      outputs = tf.matmul(inputs, weights)
      with scopes.arg_scope([batch_norm], is_training=is_training,
                            trainable=trainable, restore=restore):
        outputs = batch_norm(outputs, **batch_norm_params)
    else:
      bias_shape = [num_units_out,]
      bias_initializer = tf.constant_initializer(bias)
      biases = variables.variable('biases',
                                  shape=bias_shape,
                                  initializer=bias_initializer,
                                  trainable=trainable,
                                  restore=restore)
      outputs = tf.matmul(inputs, weights) + biases
    if activation:
      outputs = activation(outputs)
    return outputs


def one_hot_encoding(labels, num_classes, scope=None):
  """Transform numeric labels into onehot_labels.

  Args:
    labels: [batch_size] target labels.
    num_classes: total number of classes.
    scope: Optional scope for name_scope.
  Returns:
    one hot encoding of the labels.
  """
  with tf.compat.v1.name_scope(scope, 'OneHotEncoding', [labels]):
    batch_size = labels.get_shape()[0]
    indices = tf.expand_dims(tf.range(0, batch_size), 1)
    labels = tf.cast(tf.expand_dims(labels, 1), indices.dtype)
    concated = tf.concat([indices, labels], 1)
    onehot_labels = tf.sparse_to_dense(
        concated, tf.pack([batch_size, num_classes]), 1.0, 0.0)
    onehot_labels.set_shape([batch_size, num_classes])
    return onehot_labels


@scopes.add_arg_scope
def max_pool(inputs, kernel_size, stride=2, padding='VALID', scope=None):
  """Adds a Max Pooling layer.

  It is assumed by the wrapper that the pooling is only done per image and not
  in depth or batch.

  Args:
    inputs: a tensor of size [batch_size, height, width, depth].
    kernel_size: a list of length 2: [kernel_height, kernel_width] of the
      pooling kernel over which the op is computed. Can be an int if both
      values are the same.
    stride: a list of length 2: [stride_height, stride_width].
      Can be an int if both strides are the same.  Note that presently
      both strides must have the same value.
    padding: the padding method, either 'VALID' or 'SAME'.
    scope: Optional scope for name_scope.

  Returns:
    a tensor representing the results of the pooling operation.
  Raises:
    ValueError: if 'kernel_size' is not a 2-D list
  """
  with tf.compat.v1.name_scope(scope, 'MaxPool', [inputs]):
    kernel_h, kernel_w = _two_element_tuple(kernel_size)
    stride_h, stride_w = _two_element_tuple(stride)
    return tf.nn.max_pool(inputs,
                          ksize=[1, kernel_h, kernel_w, 1],
                          strides=[1, stride_h, stride_w, 1],
                          padding=padding)


@scopes.add_arg_scope
def avg_pool(inputs, kernel_size, stride=2, padding='VALID', scope=None):
  """Adds a Avg Pooling layer.

  It is assumed by the wrapper that the pooling is only done per image and not
  in depth or batch.

  Args:
    inputs: a tensor of size [batch_size, height, width, depth].
    kernel_size: a list of length 2: [kernel_height, kernel_width] of the
      pooling kernel over which the op is computed. Can be an int if both
      values are the same.
    stride: a list of length 2: [stride_height, stride_width].
      Can be an int if both strides are the same.  Note that presently
      both strides must have the same value.
    padding: the padding method, either 'VALID' or 'SAME'.
    scope: Optional scope for name_scope.

  Returns:
    a tensor representing the results of the pooling operation.
  """
  with tf.compat.v1.name_scope(scope, 'AvgPool', [inputs]):
    kernel_h, kernel_w = _two_element_tuple(kernel_size)
    stride_h, stride_w = _two_element_tuple(stride)
    return tf.nn.avg_pool(inputs,
                          ksize=[1, kernel_h, kernel_w, 1],
                          strides=[1, stride_h, stride_w, 1],
                          padding=padding)


@scopes.add_arg_scope
def dropout(inputs, keep_prob=0.5, is_training=True, scope=None):
  """Returns a dropout layer applied to the input.

  Args:
    inputs: the tensor to pass to the Dropout layer.
    keep_prob: the probability of keeping each input unit.
    is_training: whether or not the model is in training mode. If so, dropout is
    applied and values scaled. Otherwise, inputs is returned.
    scope: Optional scope for name_scope.

  Returns:
    a tensor representing the output of the operation.
  """
  if is_training and keep_prob > 0:
    with tf.name_scope(scope, 'Dropout', [inputs]):
      return tf.nn.dropout(inputs, keep_prob)
  else:
    return inputs


def flatten(inputs, scope=None):
  """Flattens the input while maintaining the batch_size.

    Assumes that the first dimension represents the batch.

  Args:
    inputs: a tensor of size [batch_size, ...].
    scope: Optional scope for name_scope.

  Returns:
    a flattened tensor with shape [batch_size, k].
  Raises:
    ValueError: if inputs.shape is wrong.
  """
  if len(inputs.get_shape()) < 2:
    raise ValueError('Inputs must be have a least 2 dimensions')
  dims = inputs.get_shape()[1:]
  k = dims.num_elements()
  with tf.compat.v1.name_scope(scope, 'Flatten', [inputs]):
    return tf.reshape(inputs, [-1, k])


def repeat_op(repetitions, inputs, op, *args, **kwargs):
  """Build a sequential Tower starting from inputs by using an op repeatedly.

  It creates new scopes for each operation by increasing the counter.
  Example: given repeat_op(3, _, ops.conv2d, 64, [3, 3], scope='conv1')
    it will repeat the given op under the following variable_scopes:
      conv1/Conv
      conv1/Conv_1
      conv1/Conv_2

  Args:
    repetitions: number or repetitions.
    inputs: a tensor of size [batch_size, height, width, channels].
    op: an operation.
    *args: args for the op.
    **kwargs: kwargs for the op.

  Returns:
    a tensor result of applying the operation op, num times.
  Raises:
    ValueError: if the op is unknown or wrong.
  """
  scope = kwargs.pop('scope', None)
  with tf.compat.v1.variable_scope(scope, 'RepeatOp', [inputs]):
    tower = inputs
    for _ in range(repetitions):
      tower = op(tower, *args, **kwargs)
    return tower
