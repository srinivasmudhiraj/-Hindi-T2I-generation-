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
"""Contains the new arg_scope used for TF-Slim ops.

  Allows one to define models much more compactly by eliminating boilerplate
  code. This is accomplished through the use of argument scoping (arg_scope).

  Example of how to use scopes.arg_scope:

  with scopes.arg_scope(ops.conv2d, padding='SAME',
                      stddev=0.01, weight_decay=0.0005):
    net = ops.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
    net = ops.conv2d(net, 256, [5, 5], scope='conv2')

  The first call to conv2d will overwrite padding:
    ops.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
              stddev=0.01, weight_decay=0.0005, scope='conv1')

  The second call to Conv will use predefined args:
    ops.conv2d(inputs, 256, [5, 5], padding='SAME',
               stddev=0.01, weight_decay=0.0005, scope='conv2')

  Example of how to reuse an arg_scope:
  with scopes.arg_scope(ops.conv2d, padding='SAME',
                      stddev=0.01, weight_decay=0.0005) as conv2d_arg_scope:
    net = ops.conv2d(net, 256, [5, 5], scope='conv1')
    ....

  with scopes.arg_scope(conv2d_arg_scope):
    net = ops.conv2d(net, 256, [5, 5], scope='conv2')

  Example of how to use scopes.add_arg_scope:

  @scopes.add_arg_scope
  def conv2d(*args, **kwargs)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import functools

from tensorflow.python.framework import ops
# Legacy TensorFlow compatibility
import tensorflow as tf
ops = tf.compat.v1
# Constants for scope
_ARGSTACK_KEY = ("__arg_stack",)
_DECORATED_OPS = set()


def _get_arg_stack():
    """Gets the current argument stack from TensorFlow's ops collection."""
    stack = ops.get_collection(_ARGSTACK_KEY)
    if stack:
        return stack[0]
    else:
        # If no stack exists, initialize a new stack
        stack = [{}]
        ops.add_to_collection(_ARGSTACK_KEY, stack)
        return stack


def _current_arg_scope():
    """Returns the current scope dictionary."""
    stack = _get_arg_stack()
    return stack[-1]


def _add_op(op):
    """Registers an operation into the global set of decorated ops."""
    key_op = (op.__module__, op.__name__)
    if key_op not in _DECORATED_OPS:
        _DECORATED_OPS.add(key_op)


@contextlib.contextmanager
def arg_scope(list_ops_or_scope, **kwargs):
    """
    Sets defaults for the specified operations.

    Args:
        list_ops_or_scope: A list/tuple of ops to apply the defaults to or a dictionary representing reused scope.
        **kwargs: Key-value pairs of arguments to apply to the operations in `list_ops`.
    """
    if isinstance(list_ops_or_scope, dict):
        # Reuse an existing scope
        if kwargs:
            raise ValueError("When reusing a scope, kwargs must be empty.")
        current_scope = list_ops_or_scope.copy()
        try:
            _get_arg_stack().append(current_scope)
            yield current_scope
        finally:
            _get_arg_stack().pop()
    else:
        if not isinstance(list_ops_or_scope, (list, tuple)):
            raise TypeError("Expected list or tuple of operations to scope, or a reused dictionary scope.")

        try:
            current_scope = _current_arg_scope().copy()
            for op in list_ops_or_scope:
                key_op = (op.__module__, op.__name__)
                if not has_arg_scope(op):
                    raise ValueError(f"{key_op} is not decorated with @add_arg_scope")
                if key_op in current_scope:
                    current_kwargs = current_scope[key_op].copy()
                    current_kwargs.update(kwargs)
                    current_scope[key_op] = current_kwargs
                else:
                    current_scope[key_op] = kwargs.copy()
            _get_arg_stack().append(current_scope)
            yield current_scope
        finally:
            _get_arg_stack().pop()


def add_arg_scope(func):
    """Decorates a function with args so it can be used within an arg_scope.

    Args:
        func: function to decorate.

    Returns:
        A decorated function that uses only known args from the current scope.
    """
    @functools.wraps(func)
    def func_with_args(*args, **kwargs):
        # Retrieve the current scope for the operation defaults
        current_scope = _current_arg_scope()
        key_func = (func.__module__, func.__name__)
        
        # If this function is already in the current scope, merge only its expected arguments
        if key_func in current_scope:
            scope_defaults = current_scope[key_func]
            # Only use arguments expected by the function's signature
            filtered_args = {k: v for k, v in scope_defaults.items() if k in func.__code__.co_varnames}
            # Update with any explicitly provided kwargs
            filtered_args.update({k: v for k, v in kwargs.items() if k in func.__code__.co_varnames})
            kwargs = filtered_args

        return func(*args, **kwargs)
    
    # Register the function into the decorator ops tracking mechanism
    _add_op(func)
    return func_with_args

def has_arg_scope(func):
    """Checks if a function is decorated with @add_arg_scope."""
    key_op = (func.__module__, func.__name__)
    return key_op in _DECORATED_OPS


# Define a custom conv2d to integrate defaults via scoping
@add_arg_scope
def conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME', scope=None, **kwargs):
    """
    Custom implementation for a Conv2D operation with defaults injected dynamically.

    Args:
        inputs: Input tensor to pass through convolution.
        num_outputs: Number of output channels for convolution.
        kernel_size: Size of the kernel.
        stride: The convolution stride.
        padding: Padding type for the convolution.
        scope: Variable scope name.
        **kwargs: Allow additional keyword arguments from argument scoping.

    Returns:
        Output tensor after convolution.
    """
    # Handle passed defaults from kwargs
    activation_fn = kwargs.get('activation_fn', tf.nn.relu)  # Default to ReLU if not specified
    
    with tf.compat.v1.variable_scope(scope or 'conv2d'):
        # Perform convolution with given or default activation function
        conv = tf.compat.v1.layers.conv2d(
            inputs=inputs,
            filters=num_outputs,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            activation=activation_fn,  # Dynamically set based on scope defaults
        )
    return conv
