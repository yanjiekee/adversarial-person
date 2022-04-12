r"""Adversarial patch generation, transformation, and application"""

import tensorflow as tf

def init(height=100, width=100, random=False):
  """Initialise a tf.Variable with shape (1, height, width, 3)"""

  if random is True:
    random = tf.random.Generator.from_seed(1)
    adversarial_patch = random.uniform(shape=(1, height, width, 3), minval=0, maxval=255, dtype=tf.float32)
    adversarial_patch = tf.Variable(adversarial_patch, dtype=tf.float32)
  else:
    adversarial_patch = tf.Variable(tf.constant(127.0, shape=[1, height, width, 3], dtype=tf.float32), dtype=tf.float32)

  return adversarial_patch

def transform(box, patch, mask_width=640, random_size=False, random_location=False):
  """
  Generate an adversarial patch mask, with the condition that patch size is
  smaller than the box size

  Argument:
    box - A bounding boxes normalised coordinates with shape (1, 4)
    patch - A tf.Variable training adversarial patch of shape (1, None, None, 3)

  Return:
    An adversarial patch mask of shape (1, 640, 640, 3) where irrelevant spaces are
    occupied with constant -1
  """
  # Fudge patch such that there're no black [0, 0, 0] pixel
  patch = tf.add(patch, 1)

  #TODO: Environment transformation of the adversarial patch

  # Get box information
  box = tf.squeeze(box)
  denormalised_box = tf.math.round(tf.multiply(box, mask_width))

  ymin, xmin, ymax, xmax = tf.unstack(denormalised_box, axis=0)
  box_height = ymax - ymin
  box_width = xmax - xmin

  # Resize patch to be 0.4 - 0.6 factor smaller than the box
  if random_size is True:
    patch_to_box_factor = tf.cast(tf.squeeze(tf.random.uniform(shape=[1], minval=0.4, maxval=0.6, dtype=tf.float32)), dtype=tf.float32)
  else:
    patch_to_box_factor = tf.constant(0.5, tf.float32)

  patch_height = tf.multiply(tf.where(box_height > box_width, box_width, box_height), patch_to_box_factor)
  patch_width = patch_height

  patch = tf.squeeze(patch)
  patch = tf.image.resize(patch, size=[patch_height, patch_width])

  # Create an imaginary box for the top-left corner of the patch
  ymin_i = ymin
  xmin_i = xmin
  ymax_i = ymax - patch_height
  xmax_i = xmax - patch_width

  imaginary_box = tf.clip_by_value(tf.stack([ymin_i, xmin_i, ymax_i, xmax_i], axis=0), clip_value_min=0, clip_value_max=mask_width)
  imaginary_box_height = ymax_i - ymin_i
  imaginary_box_width = xmax_i - xmin_i

  if random_location is True:
    # Randomly pick a location on a person
    yloc = tf.squeeze(tf.random.uniform(shape=[1], minval=0, maxval=1, dtype=tf.float32))
    xloc = tf.squeeze(tf.random.uniform(shape=[1], minval=0, maxval=1, dtype=tf.float32))
  else:
    yloc = tf.constant(0.5, tf.float32)
    xloc = tf.constant(0.5, tf.float32)

  ystart = tf.cast(tf.clip_by_value(ymin + tf.math.round(imaginary_box_height * yloc), 0, mask_width-patch_height), dtype=tf.int32)
  xstart = tf.cast(tf.clip_by_value(xmin + tf.math.round(imaginary_box_width * xloc), 0, mask_width-patch_width), dtype=tf.int32)

  # Expand the patch image such that patch is on the bounding box of a person
  transformed_patch = tf.image.pad_to_bounding_box(patch, ystart, xstart, 640, 640)

  # Padded pixel = (0, 0, 0), black pixel = (1, 1, 1)
  # Unfudge the patch such that padded pixel = (-1, -1, -1), black pixel = (0, 0, 0)
  # This will cause clipping warning when display using matplotlib
  transformed_patch = tf.subtract(transformed_patch, 1)

  return tf.expand_dims(transformed_patch, axis=0)

def apply(image, patch):
  """
  Apply patch mask onto image

  Argument:
    image - A tf.float32 image with shape (1, 640, 640, 3)
    patch - An tf.float32 adversarial patch mask of shape (1, 640, 640, 3)

  Return:
    An the combination of image and patch
  """
  patch = tf.squeeze(patch)
  image = tf.squeeze(image)
  applied_patch = tf.where(patch == -1, image, patch)

  return tf.expand_dims(tf.cast(applied_patch, tf.float32), axis=0)
