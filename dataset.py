r"""Dataset input pipeline from INRIAPerson dataset, however gt labels are generated using detection model"""

import tensorflow as tf
import numpy as np

import os
import random

from PIL import Image
from six import BytesIO

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

def tensorfy_and_resize_img(img_list_np, size):
  """Convert to tensor and resize to fit detect()'s input signature which is
  tf.float32 shape [1, 1024, 1024, 3]
  """
  img_list_ts = []
  for img in img_list_np:
    img_list_ts.append(tf.expand_dims(tf.cast(tf.image.resize_with_pad(tf.convert_to_tensor(img), size, size), tf.float32), axis=0))

  return img_list_ts

def get_label_dataset_fn(model, img_size, label_threshold=0.8):
  # Create a graph execution for faster future use
  @tf.function(input_signature=(
      tf.TensorSpec(shape=(1, img_size, img_size, 3), dtype=tf.float32),
  ))
  def model_selfdetect(input_tensor):
    preprocessed_image, shapes = model.preprocess(input_tensor)
    prediction_dict = model.predict(preprocessed_image, shapes)
    return model.postprocess(prediction_dict, shapes)

  def label_dataset_fn(img_list_ts):
    """Generate groundtruth boxes, class using detection model
    """
    box_list_np = []
    class_list_np = []

    for id, img in enumerate(img_list_ts):
      predictions = model_selfdetect(img)
      n = int(tf.squeeze(predictions['num_detections']))
      boxes = np.zeros(shape=(n,4))
      classes = np.zeros(shape=n)
      e = 0

      for i in range(n):
        if(tf.squeeze(predictions['detection_scores']).numpy()[i] > label_threshold):
          e += 1
          boxes[i] = tf.squeeze(predictions['detection_boxes']).numpy()[i]
          classes[i] = tf.squeeze(predictions['detection_classes']).numpy()[i] + 1

      box_list_np.append(boxes[0:e])
      class_list_np.append(classes[0:e])

    return box_list_np, class_list_np

  return label_dataset_fn

def filter_no_person(img_list_ts, box_list_np, class_list_np, person_class_id):
  """Remove data with no person detected
  """
  num_of_removed_item = 0
  for id in range(len(img_list_ts)):
    num_of_person = 0
    for c in class_list_np[id - num_of_removed_item]:
      if c == person_class_id:
        num_of_person += 1

    if num_of_person == 0:
      # Remove list[id]
      del img_list_ts[id - num_of_removed_item]
      del box_list_np[id - num_of_removed_item]
      del class_list_np[id - num_of_removed_item]
      num_of_removed_item += 1

  return img_list_ts, box_list_np, class_list_np

def filter_multiple_person(img_list_ts, box_list_np, class_list_np, person_class_id, max_person=1):
  """Remove data with more than one person detected
  """

  num_of_removed_item = 0
  for id in range(len(img_list_ts)):
    num_of_person = 0
    for c in class_list_np[id - num_of_removed_item]:
      if c == person_class_id:
        num_of_person += 1
        if num_of_person > max_person:
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

def filter_excessive_detection(img_list_ts, box_list_np, class_list_np, max_detections=4):
  """Remove data with excessive detection, including non-person class
  This is a workaround in model.provide_groundtruth() function where function
  cannot receive empty list as argument

  This is to reduce the number of retracing during training phase
  """
  if max_detections < 0:
    raise Exception("Maximum detection cannot be smaller than zero")

  num_of_removed_item = 0
  for id in range(len(img_list_ts)):
    if len(class_list_np[id - num_of_removed_item]) > max_detections:
      # Remove list[id]
      del img_list_ts[id - num_of_removed_item]
      del box_list_np[id - num_of_removed_item]
      del class_list_np[id - num_of_removed_item]
      num_of_removed_item += 1
      break

  return img_list_ts, box_list_np, class_list_np

def _calc_iou(box1, box2):
  """Calculate the Intersection over Union between two bounding box
  """
  ymin1, xmin1, ymax1, xmax1 = np.moveaxis(box1 * 100, 0, 0)
  ymin2, xmin2, ymax2, xmax2 = np.moveaxis(box2 * 100, 0, 0)

  x_overlap = max(0, (min(xmax1, xmax2) - max(xmin1, xmin2)))
  y_overlap = max(0, (min(ymax1, ymax2) - max(ymin1, ymin2)))
  area_intersect = x_overlap * y_overlap

  area_box1 = (xmax1 - xmin1) * (ymax1 - ymin1)
  area_box2 = (xmax2 - xmin2) * (ymax2 - ymin2)
  area_union = area_box1 + area_box2 - area_intersect * 2

  iou = area_intersect / area_union
  return iou

def filter_large_iou(img_list_ts, box_list_np, class_list_np, adv_box_list_np, iou_threshold=0.25):
  """Remove data that has large IOU between the adversarial person class and any other detection(s)
  """

  num_of_removed_item = 0
  for id in range(len(img_list_ts)):
    for b in box_list_np[id - num_of_removed_item]:
      iou = _calc_iou(adv_box_list_np[id - num_of_removed_item], b)
      if iou > iou_threshold:
        # Remove list[id]
        del img_list_ts[id - num_of_removed_item]
        del box_list_np[id - num_of_removed_item]
        del class_list_np[id - num_of_removed_item]
        del adv_box_list_np[id - num_of_removed_item]
        num_of_removed_item += 1
        break

  return img_list_ts, box_list_np, class_list_np, adv_box_list_np

def generate_mallicious_objectness_gt(box_list_np, class_list_np, person_class_id):
  """Select the person class with highest confidence and remove it from
  the dataset list, return the detection's bounding boxes to generate
  adversarial image
  """
  adv_box_list_np = []
  mal_box_list_np = []
  mal_class_list_np = []

  for i in range(len(box_list_np)):
    for j in range(len(class_list_np[i])):
      if class_list_np[i][j] == person_class_id:
        adv_box_list_np.append(box_list_np[i][j])
        mal_box_list_np.append(np.delete(box_list_np[i], j, axis=0))
        mal_class_list_np.append(np.delete(class_list_np[i], j))
        break

  return mal_box_list_np, mal_class_list_np, adv_box_list_np

def generate_mallicious_classification_gt(box_list_np, class_list_np, person_class_id, mallicious_class_id):
  """Select the person class with highest confidence and modify its class to
  a mallicious class, return the detection's bounding boxes to generate
  adversarial image
  """
  adv_box_list_np = []
  mal_class_list_np = class_list_np

  for i in range(len(box_list_np)):
    for j in range(len(mal_class_list_np[i])):
      if mal_class_list_np[i][j] == person_class_id:
        adv_box_list_np.append(box_list_np[i][j])
        mal_class_list_np[i][j] = mallicious_class_id
        break

  return mal_class_list_np, adv_box_list_np

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

def split(img_list, box_list, class_list, adv_box_list, split):
  """Split dataset according to the split ratio where "b" is the split size
  """
  full_size = len(adv_box_list)
  split_size = round(full_size * split)

  all_keys = list(range(full_size))
  random.shuffle(all_keys)
  keys_a = all_keys[split_size:]
  keys_b = all_keys[:split_size]

  img_list_a      = [img_list[key] for key in keys_a]
  box_list_a      = [box_list[key] for key in keys_a]
  class_list_a    = [class_list[key] for key in keys_a]
  adv_box_list_a  = [adv_box_list[key] for key in keys_a]

  img_list_b     = [img_list[key] for key in keys_b]
  box_list_b     = [box_list[key] for key in keys_b]
  class_list_b   = [class_list[key] for key in keys_b]
  adv_box_list_b = [adv_box_list[key] for key in keys_b]

  return img_list_a, box_list_a, class_list_a, adv_box_list_a, img_list_b, box_list_b, class_list_b, adv_box_list_b

def batch(img_list, box_list, class_list, adv_box_list, batch_size):
  """Sample a batch of dataset randomly according to the batch ratio
  """
  full_size = len(adv_box_list)
  if batch_size > full_size:
    raise Exception("Batch size is larger than dataset size")

  all_keys = list(range(full_size))
  random.shuffle(all_keys)
  keys = all_keys[:batch_size]

  img_batch      = [img_list[key] for key in keys]
  box_batch      = [box_list[key] for key in keys]
  class_batch    = [class_list[key] for key in keys]
  adv_box_batch  = [adv_box_list[key] for key in keys]

  return img_batch, box_batch, class_batch, adv_box_batch
