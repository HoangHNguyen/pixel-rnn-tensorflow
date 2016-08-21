import numpy as np
import re
import scipy.misc
from skimage.transform import resize
import matplotlib.pyplot as plt
from random import shuffle
from enum import Enum

class loc_mode(Enum):
  random = 0
  center = 1
  percent = 2

def standardize(img, std_h):
  cur_h, cur_w = img.shape
  print("Original size = {}".format(img.shape))
  woh = float(cur_w / cur_h)
  std_w = int(woh * std_h)
  std_img = resize(img, (std_h, std_w))
  #~ std_img = np.resize(img, (std_h, std_w))  
  print("Std size = {}".format(std_img.shape))
  return std_img

def copyMakeBorder(img, left, right, top, bottom, mode = 1):
  cur_h, cur_w = img.shape
  
  new_h = cur_h + top + bottom
  new_w = cur_w + left + right
  
  copy_img = zeros((new_h, new_w))
  
  new_x = 0
  new_y = 0
  
  if mode == 1:
    new_x = new_w / 2 - cur_w / 2
    new_y = new_h / 2 - cur_h / 2
    
  # Re-adjust new_x and new_y to avoid out-of-range
  # new_x + cur_w = new_w + a
  # a = new_x + cur_w - new_w
  # update_x = new_x - a = new_x - new_x - cur_w + new_w = new_w - cur_w
  if new_x + cur_w > new_w:
    new_x = new_w - cur_w
  if new_y + cur_h > new_h:
    new_y = new_h - cur_h
  
  copy_img[new_y:new_y+cur_h-1][new_x:new_x+new_w-1] = img
  return copy_img
  

def thresholding(image, th):
  return image < th

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
    #~ Black on white
    
    line_image = thresholding(line_image, thres)
    line_image = standardize(line_image, std_h = 32)
    list_images.append(line_image)
    list_labels.append(line_text)
    #~ print(line_text)
    #~ print(line_image.shape)
    plt.imshow(line_image,).set_cmap('gray')
    plt.show()       

    ind = ind + 1
    if ind == 1:
      break
    
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
