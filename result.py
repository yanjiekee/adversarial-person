import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import os

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
from google.colab import drive

def init_drive():
  """Mount google drive
  """
  # Mount google drive
  if not os.path.isdir("/content/drive"):
    drive.mount("/content/drive")

def init():
  # Create dir to store results in drive
  drive_result_dir = "/content/drive/MyDrive/Adversarial"
  if not os.path.isdir(drive_result_dir):
    os.mkdir(drive_result_dir)

  drive_result_dir = "/content/drive/MyDrive/Adversarial/result"
  if not os.path.isdir(drive_result_dir):
    os.mkdir(drive_result_dir)

  return drive_result_dir

def new_record(drive_dir):
    """Create new record folder in the dir with date & time
    """
    ct = datetime.datetime.now()
    filename = f"{ct.year}-{ct.month}-{ct.day}_{ct.hour}:{ct.minute}:{ct.second}"
    result_dir = os.path.join(drive_dir, filename)
    if not os.path.isdir(result_dir):
      os.mkdir(result_dir)

    return result_dir

def store_adv_checkpoint(variable, dir):
  adv_checkpoint = tf.train.Checkpoint(patch=variable)
  adv_checkpoint.write(os.path.join(dir, "adv_ckpt"))

def store_loss_history_csv(loss_history, sampling_rate, dir):
  """Store loss history in CSV format
  """
  filepath = os.path.join(dir, 'loss_history.csv')
  length = len(loss_history)
  stop = length * sampling_rate

  x = np.arange(start=0, stop=stop, step=sampling_rate)
  y = loss_history

  plot = np.column_stack((x, y))

  df = pd.DataFrame(plot)
  df.to_csv(filepath, index=False)

  return filepath

def retrieve_loss_history_csv(filepath):
  """Retrieve CSV file to loss history list and loss sampling rate
  """
  df = pd.read_csv(filepath)
  x, y = np.moveaxis(df.to_numpy(), source=1, destination=0)

  sampling_rate = x[1] / y[1]
  loss_history = y

  return loss_history, sampling_rate

def print_loss_history_plot(loss_history, sampling_rate, size=None):
  """Print the loss history plot
  """
  length = len(loss_history)
  stop = length * sampling_rate

  x = np.arange(start=0, stop=stop, step=sampling_rate).astype(int)
  y = loss_history

  fig, ax = plt.subplots(figsize=size)
  ax.plot(x, y, '.-')
  ax.title.set_text("Loss v Iteration Count")
  ax.xaxis.set_major_locator(MaxNLocator(integer=True))

def print_and_save_loss_history_plot(loss_history, sampling_rate, dir, size=None):
  """Print the loss history plot and save it as PNG
  """
  length = len(loss_history)
  stop = length * sampling_rate

  x = np.arange(start=0, stop=stop, step=sampling_rate).astype(int)
  y = loss_history

  fig, ax = plt.subplots(figsize=size)
  ax.plot(x, y, '.-')
  ax.title.set_text("Loss v Iteration Count")
  ax.xaxis.set_major_locator(MaxNLocator(integer=True))
  fig.savefig(os.path.join(dir, "loss_history.png"))
