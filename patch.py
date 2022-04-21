r"""Adversarial patch generation, transformation, and application"""

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

def init(height=100, width=100, random=False, random_seed=14):
  """Return a tf.constant with shape (1, height, width, 3) with value [0, 1]
  """

  if random is True:
    random = tf.random.Generator.from_seed(random_seed)
    adversarial_patch = random.uniform(shape=(1, height, width, 3), minval=0, maxval=1, dtype=tf.float32)
  else:
    adversarial_patch = tf.constant(0.5, shape=[1, height, width, 3], dtype=tf.float32)

  return adversarial_patch

def show(patch):
  """Print adversarial patch using matplotlib"""
  squeezed_patch = tf.squeeze(patch)
  plt.imshow(squeezed_patch.numpy())

@tf.function(input_signature=(
    tf.TensorSpec(shape=[4], dtype=tf.float32),
    tf.TensorSpec(shape=[1, 100, 100, 3], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.bool),
    tf.TensorSpec(shape=[], dtype=tf.bool)
))
def transform(box, patch,
              mask_width=tf.constant(1024, tf.float32),
              random_size=tf.constant(False),
              random_location=tf.constant(False)):
  """
  Generate an adversarial patch mask

  Argument:
    box - A bounding boxes normalised coordinates of shape (4)
    patch - A tf.Variable training adversarial patch of shape (1, 100, 100, 3) with range [0, 1]

  Return:
    An adversarial patch mask of shape (1, 640, 640, 3) with range [0, 255], and where irrelevant spaces are occupied with constant -1
  """
  # Convert patch to range [0, 255], then fudge the patch by 1 such that the black pixel is (1, 1, 1) instead of (0, 0, 0)
  patch = tf.add(tf.math.round(tf.multiply(tf.squeeze(patch), 255.0)), 1.0)

  # Get box information
  denormalised_box = tf.math.round(tf.multiply(box, mask_width))

  ymin, xmin, ymax, xmax = tf.unstack(denormalised_box, axis=0)
  box_height = ymax - ymin
  box_width = xmax - xmin

  # Resize patch to be 0.4 - 0.6 factor smaller than the box
  if random_size is True:
    patch_to_box_factor = tf.cast(tf.squeeze(tf.random.uniform(shape=[1], minval=0.4, maxval=0.6, dtype=tf.float32)), dtype=tf.float32)
  else:
    patch_to_box_factor = tf.constant(0.5, tf.float32)

  patch_width = tf.math.round(tf.multiply(tf.where(box_height > box_width, box_width, box_height), patch_to_box_factor))
  patch = tf.image.resize(patch, size=[patch_width, patch_width])

  # Create an imaginary box for the top-left corner of the patch
  imaginary_box_height = box_height - patch_width
  imaginary_box_width = box_width - patch_width

  if random_location is True:
    # Randomly pick a location on a person
    yloc = tf.squeeze(tf.random.uniform(shape=[1], minval=0.4, maxval=0.6, dtype=tf.float32))
    xloc = tf.squeeze(tf.random.uniform(shape=[1], minval=0.4, maxval=0.6, dtype=tf.float32))
  else:
    yloc = tf.constant(0.5, tf.float32)
    xloc = tf.constant(0.5, tf.float32)

  ystart = tf.cast(tf.clip_by_value(ymin + tf.math.round(imaginary_box_height * yloc), 0, mask_width-patch_width), tf.int32)
  xstart = tf.cast(tf.clip_by_value(xmin + tf.math.round(imaginary_box_width * xloc), 0, mask_width-patch_width), tf.int32)
  mask_width = tf.cast(mask_width, tf.int32)

  # Expand the patch image such that patch is on the bounding box of a person
  transformed_patch = tf.image.pad_to_bounding_box(patch, ystart, xstart, mask_width, mask_width)

  # Padded pixel = [0.0]
  # Fudge the actual black pixel back to (0, 0, 0) whereas the padded area are now (-1, -1, -1)
  transformed_patch = tf.subtract(transformed_patch, 1.0)

  return tf.expand_dims(transformed_patch, axis=0)

@tf.function(input_signature=(
    tf.TensorSpec(shape=[None, 1024, 1024, 3], dtype=tf.float32),
    tf.TensorSpec(shape=[None, 1024, 1024, 3], dtype=tf.float32)
))
def apply(image, patch):
  """
  Apply patch mask onto image

  Argument:
    image - A tf.float32 image with shape (batch_size, 640, 640, 3)
    patch - An tf.float32 adversarial patch mask of shape (batch_size, 640, 640, 3)

  Return:
    An the combination of image and patch
  """
  applied_patch = tf.where(patch == -1.0, image, patch)

  return applied_patch
