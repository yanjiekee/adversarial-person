r"""Dataset input pipeline from INRIAPerson dataset, however gt labels are generated using detection model"""

import tensorflow as tf
import numpy as np

import os
import random

from PIL import Image
from six import BytesIO

CATEGORY_PERSON_CLASS_ID = 1
CATEGORY_LABEL_ID_OFFSET = 1

def _load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path.

  Returns:
    uint8 numpy array with shape [img_height, img_width, 3]
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3))

def fetch(img_dir):
  """Convert image dataset into Python lists of images

  Return
    img_list_np: A list of np.uint8 array with shape [Height, Width, 3]
  """
  img_list_np = []
  directory = os.fsencode(img_dir)
  for file in os.listdir(directory):
    filename = os.fsdecode(file)
    file_path = img_dir + '/' + filename
    img_list_np.append(_load_image_into_numpy_array(file_path))

  return img_list_np

def tensorfy(img_list_np, size):
  """Convert to tensor and resize to fit detect()'s input signature which is
  tf.float32 shape [1, 1024, 1024, 3]
  """
  img_list_ts = []
  for img in img_list_np:
    img_list_ts.append(tf.expand_dims(tf.image.resize_with_pad(tf.convert_to_tensor(img, tf.float32), size, size), axis=0))

  return img_list_ts

def get_label_fn(model, img_size):

  # Create a graph execution for faster future use
  @tf.function(input_signature=(
      tf.TensorSpec(shape=(1, img_size, img_size, 3), dtype=tf.float32),
  ))
  def detect(input_tensor):
    preprocessed_image, shapes = model.preprocess(input_tensor)
    prediction_dict = model.predict(preprocessed_image, shapes)
    return model.postprocess(prediction_dict, shapes)

  def label(img_list_ts):
    """Generate groundtruth boxes, class using detection model
    """
    box_list_np = []
    class_list_np = []

    for id, img in enumerate(img_list_ts):
      predictions = detect(img)
      n = int(tf.squeeze(predictions['num_detections']))
      boxes = np.zeros(shape=(n,4))
      classes = np.zeros(shape=n)
      e = 0

      for i in range(n):
        if(tf.squeeze(predictions['detection_scores']).numpy()[i] > 0.8):
          e += 1
          boxes[i] = tf.squeeze(predictions['detection_boxes']).numpy()[i]
          classes[i] = tf.squeeze(predictions['detection_classes']).numpy()[i] + 1

      box_list_np.append(boxes[0:e])
      class_list_np.append(classes[0:e])

    return box_list_np, class_list_np

  return label

def filter_no_person(img_list_ts, box_list_np, class_list_np):
  """Remove data with no person detected
  """

  num_of_removed_item = 0
  for id in range(len(img_list_ts)):
    num_of_person = 0
    for c in class_list_np[id - num_of_removed_item]:
      if c == CATEGORY_PERSON_CLASS_ID:
        num_of_person += 1

    if num_of_person == 0:
      # Remove list[id]
      del img_list_ts[id - num_of_removed_item]
      del box_list_np[id - num_of_removed_item]
      del class_list_np[id - num_of_removed_item]
      num_of_removed_item += 1

  return img_list_ts, box_list_np, class_list_np

def filter_multiple_person(img_list_ts, box_list_np, class_list_np):
  """Remove data with more than one person detected
  """

  num_of_removed_item = 0
  for id in range(len(img_list_ts)):
    num_of_person = 0
    for c in class_list_np[id - num_of_removed_item]:
      if c == CATEGORY_PERSON_CLASS_ID:
        num_of_person += 1
        if num_of_person > 1:
          # Remove list[id]
          del img_list_ts[id - num_of_removed_item]
          del box_list_np[id - num_of_removed_item]
          del class_list_np[id - num_of_removed_item]
          num_of_removed_item += 1
          break

  return img_list_ts, box_list_np, class_list_np

def filter_single_detection(img_list_ts, box_list_np, class_list_np):
  """Remove data with only one detection, including non-person class
  This is a workaround in model.provide_groundtruth() function where function
  cannot receive empty list as argument
  """

  num_of_removed_item = 0
  for id in range(len(img_list_ts)):
    if len(class_list_np[id - num_of_removed_item]) == 1:
      # Remove list[id]
      del img_list_ts[id - num_of_removed_item]
      del box_list_np[id - num_of_removed_item]
      del class_list_np[id - num_of_removed_item]
      num_of_removed_item += 1
      break

  return img_list_ts, box_list_np, class_list_np

def generate_mallicious_gt(box_list_np, class_list_np):
  """Select the person class with highest confidence and remove it from
  the dataset list, return the detection's bounding boxes to generate
  adversarial image
  """
  CATEGORY_INDEX_PERSON = 1

  adv_box_list_np = []

  for i in range(len(box_list_np)):
    for j in range(len(class_list_np[i])):
      if class_list_np[i][j] == CATEGORY_INDEX_PERSON:
        adv_box_list_np.append(box_list_np[i][j])
        box_list_np[i] = np.delete(box_list_np[i], j, axis=0)
        class_list_np[i] = np.delete(class_list_np[i], j)
        break

  return box_list_np, class_list_np, adv_box_list_np

def tensorfy_gt(box_list_np, class_list_np, adv_box_list_np, num_of_category):
  """Convert numpy list to tensor, including the one-hot class list
  """
  box_list_ts = []
  class_list_one_hot_ts = []
  adv_box_list_ts = []

  for box_np, class_np, adv_box_np in zip(box_list_np, class_list_np, adv_box_list_np):
    box_list_ts.append(tf.convert_to_tensor(box_np, tf.float32))
    adv_box_list_ts.append(tf.convert_to_tensor(adv_box_np, tf.float32))

    zero_indexed_class_ts = tf.convert_to_tensor(np.subtract(class_np, 1), dtype=tf.uint8)
    class_list_one_hot_ts.append(tf.one_hot(zero_indexed_class_ts, num_of_category, dtype=tf.float32))

  return box_list_ts, class_list_one_hot_ts, adv_box_list_ts
