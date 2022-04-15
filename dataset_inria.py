r"""Dataset input pipeline from INRIAPerson dataset"""

import tensorflow as tf
import numpy as np

import os
import random

from PIL import Image
from six import BytesIO
from six.moves.urllib.request import urlopen
from bs4 import BeautifulSoup

CATEGORY_PERSON_CLASS_ID = 1
CATEGORY_INDEX = {CATEGORY_PERSON_CLASS_ID: {'id': CATEGORY_PERSON_CLASS_ID, 'name': 'person'}}
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

def _load_inriaperson_xml_to_boxes(path):
  """Load the INRIAPerson dataset annotations into list of normalised
  bounding boxes with shape [N, 4]. Coordinates: [ymin, xmin, ymax, xmax]

  Returns
    A list of np.float32 array with shape [Num of detection, 4]
  """
  with open(path, 'r') as file:
    xml_file = file.read()

  soup = BeautifulSoup(xml_file, 'xml')

  # Find the image size for normalisation
  image_height = int(soup.find('height').string)
  image_width = int(soup.find('width').string)

  bbox_xml_list = soup.find_all('bndbox')
  bbox_list = []

  for bbox_xml in bbox_xml_list:
    bbox = [int(bbox_xml.ymin.string)/image_height,
            int(bbox_xml.xmin.string)/image_width,
            int(bbox_xml.ymax.string)/image_height,
            int(bbox_xml.xmax.string)/image_width]
    bbox_list.append(bbox)

  return np.array(bbox_list)

def fetch(img_dir, gt_dir):
  """Convert INRIAPerson dataset into Python lists of images and normalised groundtruth boxes

  Return
    img_list_np: A list of np.uint8 array with shape [Height, Width, 3]
    gt_list_np: A list of np.float32 array with shape [Num of detection, 4]
  """
  img_list_np = []
  gt_list_np = []
  directory = os.fsencode(img_dir)
  for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if not filename.startswith("person_and_bike"):
      filename = filename[:-4]

      img_filepath = img_dir + '/' + filename + '.png'
      img_list_np.append(_load_image_into_numpy_array(img_filepath))

      gt_filepath = gt_dir + '/' + filename + '.xml'
      gt_list_np.append(_load_inriaperson_xml_to_boxes(gt_filepath))
      continue
    else:
      continue

  if len(img_list_np) != len(gt_list_np):
    raise Exception("Image count is not equal to its corresponding groundtruth")

  return img_list_np, gt_list_np

def filter_single_person(img_list_np, gt_list_np):
  """Remove images and grouthtruths pairs that contains more than one person class
  """

  if len(img_list_np) != len(gt_list_np):
    raise Exception("Image count is not equal to its corresponding groundtruth")

  filtered_img_list_np = []
  filtered_gt_list_np = []
  for i in range (len(gt_list_np)):
    if len(gt_list_np[i]) is 1:
      filtered_img_list_np.append(img_list_np[i])
      filtered_gt_list_np.append(gt_list_np[i])

  return filtered_img_list_np, filtered_gt_list_np

def _resize_gt_box_with_padding(x_n, width, height, new_width, new_height):
  """Resize bounding boxes coordinates with padding, function arguments are
  relative to the x-axis, can be applied to y-axis the same way"""

  # Find intermediate width, where height is resized to new_height but width
  # is not padded yet
  int_width = width * new_height / height
  # De-normalise coordinate
  x = x_n * int_width

  # Find offset after padding
  offset = (new_width - int_width) / 2

  # Normalise new coordinate
  new_x = x + offset
  new_x_n = new_x / new_width

  return new_x_n

def resize_with_pad(img_list_np, gt_list_np, new_height, new_width):
  """Resize images and groudtruth boxes pairs into new size
  """
  if len(img_list_np) != len(gt_list_np):
    raise Exception("Image count is not equal to its corresponding groundtruth")

  for i in range(len(gt_list_np)):
    # Resize groundtruth boxes given that new height = new width
    height, width, _ = np.shape(img_list_np[i])

    ratio = height/width
    new_ratio = new_height/new_width

    # Concern only x or y
    if ratio > new_ratio:
      for j in range(len(gt_list_np[i])):
        gt_list_np[i][j][1] = _resize_gt_box_with_padding(gt_list_np[i][j][1], width, new_width, height, new_height)
        gt_list_np[i][j][3] = _resize_gt_box_with_padding(gt_list_np[i][j][3], width, new_width, height, new_height)
    elif ratio < new_ratio:
      for j in range(len(gt_list_np[i])):
        gt_list_np[i][j][0] = _resize_gt_box_with_padding(gt_list_np[i][j][0], height, new_height, width, new_width)
        gt_list_np[i][j][2] = _resize_gt_box_with_padding(gt_list_np[i][j][2], height, new_height, width, new_width)

    # Resize image
    img_list_np[i] = tf.image.resize_with_pad(tf.convert_to_tensor(img_list_np[i]), new_width, new_width).numpy().astype(int)

  return img_list_np, gt_list_np

def tensorfy(img_list_np, gt_list_np, num_of_category_class):
  """Convert images and groudtruth boxes numpy list into list of tensors

  Return
    img_list_ts: A list of tf.float32 of shape [1, None, None, 3]
    gt_list_ts: A list of tf.float32 of shape [Num of detection, 4]
    class_list_ts: A list of one-hot tf.float32 of shape [Num of detection, Num of class]
  """
  img_list_ts = []
  gt_list_ts = []
  class_list_ts = []

  for img_np, gt_np in zip(img_list_np, gt_list_np):
    # A list of image tf.float32 tensors with shape [1, None, None, 3]
    img_list_ts.append(tf.expand_dims(tf.convert_to_tensor(img_np, tf.float32), axis=0))

    # A list of 2-D tf.float32 tensors with shape [num_of_dection, 4], normalised and clipped
    gt_list_ts.append(tf.convert_to_tensor(gt_np, tf.float32))

    # A list of 2-D tf.float32 one-hot tensor of shape [num_of_dection, num_classes]
    num_of_detection = gt_np.shape[0]
    zero_based_class = tf.convert_to_tensor(np.ones(shape=[num_of_detection], dtype=np.int32) - CATEGORY_LABEL_ID_OFFSET)
    class_list_ts.append(tf.one_hot(zero_based_class, num_of_category_class))

  return img_list_ts, gt_list_ts, class_list_ts
