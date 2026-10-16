"""CassaDx frozen DenseNet121 + SwinV2-Tiny baseline.

Tested project dependencies: TensorFlow 2.17.1, Keras 3.4.1,
tfswin 4.1.1, NumPy 1.26.4; Python 3.11.

Input: float32 RGB images, shape (batch, 256, 256, 3), values 0–255.
This is a paper-inspired baseline, not an exact paper reproduction.
"""

import keras
import tensorflow as tf
from keras import ops
import tfswin.window as window_module
import tfswin.winatt as attention_module
import tfswin.swin as swin_module
from tfswin import SwinTransformerV2Tiny256


def partition_for_export(inputs, height, width, window_size, num_heads,
                         qkv_mult=3, dtype=None, name=None):
    """Partition attention windows without rank-7/8 transposes."""
    windows = window_module.window_partition(
        inputs, height, width, window_size, dtype=dtype, name=name)
    head_channels = windows.shape[-1] // (qkv_mult * num_heads)
    outputs = ops.reshape(
        windows, [-1, window_size ** 2, qkv_mult, num_heads, head_channels])
    return ops.transpose(outputs, [2, 0, 3, 1, 4])


def reverse_for_export(inputs, height, width, window_size, num_heads,
                       dtype=None, name=None):
    """Merge attention windows using lower-rank transposes."""
    head_channels = inputs.shape[-1]
    outputs = ops.transpose(inputs, [0, 2, 1, 3])
    outputs = ops.reshape(
        outputs, [-1, window_size ** 2, num_heads * head_channels])
    return window_module.window_reverse(
        outputs, height, width, window_size, dtype=dtype, name=name)


def roll_for_export(inputs, shifts, axes):
    """Implement the spatial cyclic shifts used by Swin using Gather."""
    outputs = inputs
    for shift, axis in zip(shifts, axes):
        length = tf.shape(outputs)[axis]
        shift = tf.cast(shift, tf.int32)
        indices = tf.math.floormod(tf.range(length) - shift, length)
        outputs = tf.gather(outputs, indices, axis=axis)
    return outputs


class _SwinExportOps:
    """Override Roll only in tfswin.swin, not in global Keras operations."""
    _cassadx_export_proxy = True

    def __init__(self, original_ops):
        self.original_ops = original_ops

    def __getattr__(self, name):
        return getattr(self.original_ops, name)

    def roll(self, inputs, shift, axis):
        return roll_for_export(inputs, shift, axis)


def apply_export_fixes():
    """Apply session-local compatibility fixes before building the model.

    Does not edit installed package files. Repeated calls are safe.
    Changes affect subsequently built tfswin models in this process.
    """
    window_module.window_partition_fused = partition_for_export
    window_module.window_reverse_fused = reverse_for_export
    attention_module.window_partition_fused = partition_for_export
    attention_module.window_reverse_fused = reverse_for_export
    if not getattr(swin_module.ops, "_cassadx_export_proxy", False):
        swin_module.ops = _SwinExportOps(swin_module.ops)


def build_hybrid_model(seed=42, weights="imagenet"):
    """Build frozen pretrained branches with a new five-class classifier.

    Every call creates a NEW classifier; it does not restore trained weights.
    Both branches include the appropriate normalization in this graph.
    """
    apply_export_fixes()
    keras.utils.set_random_seed(seed)

    dense_branch = keras.applications.DenseNet121(
        weights=weights, include_top=False,
        input_shape=(256, 256, 3), pooling=None)
    swin_branch = SwinTransformerV2Tiny256(
        weights=weights, include_top=False,
        input_shape=(256, 256, 3), pooling=None,
        include_preprocessing=True)
    dense_branch.trainable = False
    swin_branch.trainable = False

    inputs = keras.Input(shape=(256, 256, 3), dtype="float32",
                         name="leaf_image")
    dense_maps = dense_branch(
        keras.applications.densenet.preprocess_input(inputs), training=False)
    swin_maps = swin_branch(inputs, training=False)
    dense_vector = keras.layers.GlobalAveragePooling2D(name="dense_pool")(dense_maps)
    swin_vector = keras.layers.GlobalAveragePooling2D(name="swin_pool")(swin_maps)
    combined = keras.layers.Concatenate(name="feature_fusion")(
        [dense_vector, swin_vector])
    outputs = keras.layers.Dense(
        5, activation="softmax", name="disease_classifier")(combined)
    return keras.Model(inputs, outputs, name="CassaDx_baseline")
