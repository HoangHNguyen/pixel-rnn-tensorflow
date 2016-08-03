import logging
logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")

import os
import sys
import urllib
import pprint
import tarfile
import tensorflow as tf

import datetime
import dateutil.tz
import numpy as np
import re
import scipy.misc
import matplotlib.pyplot as plt
from random import shuffle

pp = pprint.PrettyPrinter().pprint
logger = logging.getLogger(__name__)

def mprint(matrix, pivot=0.5):
  for array in matrix:
    print "".join("#" if i > pivot else " " for i in array)

def read_aim_ascii(file_name):
  lines = []
  with open(file_name, "r") as ins:    
    for l in ins:
      if not re.match("^\s*#",l):          
        lines.append(l)
  return lines

def load_aim_dataset(file_name, root_dir, test_percent = .25):
  print("Reading ascii file...")
  lines = read_aim_ascii(file_name)
  num_line = len(lines)
  print("Total lines = {}".format(num_line))
  list_images = [None]*num_line
  list_labels = [None]*num_line
  prev_image_name = ""
  ind = 0
  print("Reading all images and labels...")
  for l in lines:
    full_image_name, status, thres, num_char, x, y, w, h, line_text = parse_line(l)
    if full_image_name != prev_image_name:
      full_image = scipy.misc.imread(root_dir + "/" + full_image_name)
      prev_image_name = full_image_name
    #~ line_image = full_image[y:y+h-1][x:x+w-1]
    line_image = full_image
    list_images.append(line_image)
    list_labels.append(line_text)
    #~ print(line_text)
    #~ print(line_image.shape)
    #~ plt.imshow(full_image,).set_cmap('gray')
    #~ plt.show()       

    ind = ind + 1
    #~ if ind == 101:
      #~ break
    
  #~ Remove failed data
  del list_images[ind:]
  del list_labels[ind:]
  
  if len(list_images) != len(list_labels):
    print("Error: Loaded images and labels are NOT the same size")
    return -1
    
  num_sample = len(list_images)
  print("Loaded samples = {}".format(num_sample))
  #~ Shuffle dataset and split into training and test sets
  idx = [i for i in range(num_sample)]
  shuffle(idx)
  
  if test_percent >= 1:
    print("test_percent must less than 1")
    return -1
  
  train_percent = 1 - test_percent
  num_train = int(num_sample * train_percent)
  num_test = num_sample - num_train
  print( "Training set = {}, test set = {}".format(num_train, num_test) )
  
  print("Generating training and test sets...")
  train_X = [list_images[idx[i]] for i in range(num_train)]
  train_Y = [list_labels[idx[i]] for i in range(num_train)]
  test_X = [list_images[idx[i]] for i in range(num_train, num_train + num_test)]
  test_Y = [list_labels[idx[i]] for i in range(num_train, num_train + num_test)]
  
  print( "Training set = {}, test set = {}".format(len(train_Y), len(test_Y)) )
  return train_X, train_Y, test_X, test_Y
  

#~ "a01-000u-00 ok 154 19 408 746 1661 89 A|MOVE|to|stop|Mr.|Gaitskell|from"
def parse_line(txt):
  l = txt.split(" ")
  full_image_name = "unknown"
  status = "unknown"
  thres = -1
  num_char = -1;
  x = y = w = h = -1
  line = "unknown"
  ind = 0
  for ele in l:
    #~ print(ind)
    if ind == 0:
      parse_name_l = ele.split("-")
      folder = parse_name_l[0]
      sub_folder = parse_name_l[0] + "-" + parse_name_l[1]
      image_name = ele + ".png"
      full_image_name = folder + "/" + sub_folder + "/" + image_name
    elif ind == 1:
      status = ele
    elif ind == 2:
      thres = float(ele)
    elif ind == 3:
      num_char = int(ele)
    elif ind == 4:
      x = int(ele)
    elif ind == 5:
      y = int(ele)
    elif ind == 6:
      w = int(ele)
    elif ind == 7:
      h = int(ele)
    elif ind == 8:
      line = ele    
    ind = ind + 1
  #~ print(full_image_name + " line = " + line)
  return full_image_name, status, thres, num_char, x, y, w, h, line

def show_all_variables():
  total_count = 0
  for idx, op in enumerate(tf.trainable_variables()):
    shape = op.get_shape()
    count = np.prod(shape)
    print "[%2d] %s %s = %s" % (idx, op.name, shape, count)
    total_count += int(count)
  print "[Total] variable size: %s" % "{:,}".format(total_count)

def get_timestamp():
  now = datetime.datetime.now(dateutil.tz.tzlocal())
  return now.strftime('%Y_%m_%d_%H_%M_%S')

def binarize(images):
  return (np.random.uniform(size=images.shape) < images).astype('float32')

def save_images(images, height, width, n_row, n_col, 
      cmin=0.0, cmax=1.0, directory="./", prefix="sample"):
  images = images.reshape((n_row, n_col, height, width))
  images = images.transpose(1, 2, 0, 3)
  images = images.reshape((height * n_row, width * n_col))

  filename = '%s_%s.jpg' % (prefix, get_timestamp())
  scipy.misc.toimage(images, cmin=cmin, cmax=cmax) \
      .save(os.path.join(directory, filename))

def get_model_dir(config, exceptions=None):
  attrs = config.__dict__['__flags']
  pp(attrs)

  keys = attrs.keys()
  keys.sort()
  keys.remove('data')
  keys = ['data'] + keys

  names =[]
  for key in keys:
    # Only use useful flags
    if key not in exceptions:
      names.append("%s=%s" % (key, ",".join([str(i) for i in attrs[key]])
          if type(attrs[key]) == list else attrs[key]))
  return os.path.join('checkpoints', *names) + '/'

def preprocess_conf(conf):
  options = conf.__flags

  for option, value in options.items():
    option = option.lower()

def check_and_create_dir(directory):
  if not os.path.exists(directory):
    logger.info('Creating directory: %s' % directory)
    os.makedirs(directory)
  else:
    logger.info('Skip creating directory: %s' % directory)

def maybe_download_and_extract(dest_directory):
  """
  Download and extract the tarball from Alex's website.
  From https://github.com/tensorflow/tensorflow/blob/r0.9/tensorflow/models/image/cifar10/cifar10.py
  """
  DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)

  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)

  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
