# Copyright 2018 The AiGraph LLC, bin.bryandu@gmail.com. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions for downloading and reading Performance Graph data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import csv
import numpy as np
import base
import os
import re
import math
from datetime import datetime, date, time

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile


def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def draw_line_in_graph(new_array, image_height, image_width, grey_value1, grey_value2):

  # draw lines in graph
  previous_y1 = int(new_array[0,1])
  image = np.zeros(image_height*new_array.shape[0], dtype = np.uint8)
  for i in range(0, new_array.shape[0]):
    for j in range(0, int(new_array[i,0] + 1) ):
      image[j + i*image_height] = grey_value1
    for j in range(0, image_height):
      if ( j == new_array[i,1] ):
        if( j > previous_y1 ):
          for k in range ( previous_y1, j + 1 ) :
            image[k + i*image_height] = grey_value2
        else:
          for k in range ( j, previous_y1 + 1 ) :
            image[k + i*image_height] = grey_value2
        previous_y1 = j;

      
#  print("image shape->", image.shape)
  return image

def regulate_to_image(array, image_height, image_width, grey_value1, grey_value2):
  """ regulate to image image_height x image_width ."""

  max_value = np.amax(array)
  my_array = np.array(array)

  if( max_value != 0 ):
    new_array = np.trunc(my_array / (max_value * 1.01) * (image_height-1))
  else:
    new_array = np.trunc(my_array)
  print("new_array shape->", new_array.shape)
 
  image = draw_line_in_graph(new_array, image_height, image_width, grey_value1, grey_value2)

#  print("image shape->", image.shape)

  return image


def load_csv_with_header(filename, header_text, image_size, entries_in_file):
  """Load dataset from CSV file after the header row."""

  read_entry = int(entries_in_file / image_size )
  write_entry = 0
  with gfile.Open(filename) as csv_file:
    data_file = csv.reader(csv_file)
    header = next(data_file)
    while (header[0] != header_text):
      header = next(data_file)
      if( header[0] == 'Step:' ):
        read_entry = int(300/ int(header[1]) )
        print(filename, ' step is ', header[1], "seconds")
    data=[]
    for i in range(0, image_size*read_entry):
      row = next(data_file)

      # Double check the step / read_entry
      if ( i == 0 ):
        first_row_date = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
      elif ( i == 1 ) :
        second_row_date = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
        elapsedTime = second_row_date - first_row_date 
        read_entry = int(300 / elapsedTime.total_seconds())
#        print(filename, ': read 1 fro every ', read_entry , ' entries, step is ', elapsedTime.total_seconds(), "seconds")   

#      print(filename, ', read_entry:', read_entry , ", length of row:", len(row), ", row:", row, "write_entry:", write_entry, "image_size", image_size)   
      if( (i % read_entry) == 0 ):
        if ( len(row) == 1 ):
          col1 = 0
          col2 = 0
        elif( row[1] == ''):
          col1 = 0
        else:
          col1 = float(row[1])
          if ( len(row) < 3 ):
            col2 = 0
          elif ( row[2] ==''):
            col2 = 0
          else:
            col2 = float(row[2])
          if math.isnan(col1):
            col1=0;
          if math.isnan(col2):
            col2=0;
        data.append([col1, col2])
        write_entry = write_entry + 1
        
      if ( write_entry == image_size ):
        break;

#  print("csv File name, data length, data->", filename, len(data), data)
  return data

def load_csv_with_header_shift(filename, header_text, image_size, shift):
  """Load dataset from CSV file after the header row."""

  read_entry = 0
  write_entry = 0
  with gfile.Open(filename) as csv_file:
    data_file = csv.reader(csv_file)
    header = next(data_file)
    while (header[0] != header_text):
      header = next(data_file)
      if( header[0] == 'Step:' ):
        read_entry = int(300/ int(header[1]) )
        print(filename, ' step is ', header[1], "seconds", "read_entry:", read_entry)
    data=[]

    if ( read_entry != 5):
      print(" Require 1 minute counters for training")
      exit(1)
      
    for i in range(0, image_size*read_entry):
      row = next(data_file)
#      print("row", row, "entry", i)
      if( (i % read_entry) == shift ):
        if ( len(row) == 1 ):
          col1 = 0
          col2 = 0
        elif( row[1] == ''):
          col1 = 0
        else:
          col1 = float(row[1])
          if ( len(row) < 3 ):
            col2 = 0
          elif ( row[2] ==''):
            col2 = 0
          else:
            col2 = float(row[2])
          if math.isnan(col1):
            col1=0;
          if math.isnan(col2):
            col2=0;
        data.append([col1, col2])
        write_entry = write_entry + 1
        
      if ( write_entry == image_size ):
        break;

#  print("csv File name, data length, data->", filename, len(data), data)
  return data

def get_train_images(train_dir, image_size, shift):
  image = []
  image_num = 0
  image_outage_num = 0
  image_plateau_num = 0
  directory = train_dir + "0-normal/"
  for fnames in os.listdir(directory):  
    match = re.search(r'.csv',fnames)
    if( match ):
      print("Exact ", image_size, " entries from", fnames, "at", directory, "with shift", shift)   
      array = load_csv_with_header_shift(directory + fnames, 'Date', image_size, shift) 
      max_value = np.amax(array)
      my_array = np.array(array)
      if( max_value != 0 ):
        new_array = np.ceil(my_array / max_value * (image_size-2))
      else:
        new_array = np.trunc(my_array)
      image0 = draw_line_in_graph(new_array, image_size, image_size, 100, 200)
      image = np.append(image, image0)
      image_num = image_num + 1
  directory = train_dir + "1-outage/"
  for fnames in os.listdir(directory):  
    match = re.search(r'.csv',fnames)
    if( match ):
      print("Exact ", image_size, " entries from", fnames, "at", directory, "with shift", shift)   
      array = load_csv_with_header_shift(directory + fnames, 'Date', image_size, shift) 
      max_value = np.amax(array)
      my_array = np.array(array)
      if( max_value != 0 ):
        new_array = np.ceil(my_array / max_value * (image_size-2))
      else:
        new_array = np.trunc(my_array)
      image0 = draw_line_in_graph(new_array, image_size, image_size, 100, 200)
      image = np.append(image, image0)
      image_outage_num = image_outage_num + 1
  directory = train_dir + "2-plateau/"
  for fnames in os.listdir(directory):  
    match = re.search(r'.csv',fnames)
    if( match ):
      print("Exact ", image_size, " entries from", fnames, "at", directory, "with shift", shift)   
      array = load_csv_with_header_shift(directory + fnames, 'Date', image_size, shift) 
      max_value = np.amax(array)
      my_array = np.array(array)
      if( max_value != 0 ):
        new_array = np.ceil(my_array / max_value * (image_size-2))
      else:
        new_array = np.trunc(my_array)
      image0 = draw_line_in_graph(new_array, image_size, image_size, 100, 200)
      image = np.append(image, image0)
      image_plateau_num = image_plateau_num + 1

  return image, image_num, image_outage_num, image_plateau_num

def extract_images_csv(num_images, f, image_size):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
  Args:
    f: A file object with .txt, .csv.
  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].
  """

  print('Reading...', f.name)

  # Define how the title line in csv file and parse csv file
  csv_header ='Date'
  array = load_csv_with_header(f.name, csv_header, image_size, 1440)
  
  # Put array into image format
  data =regulate_to_image(array, image_size, image_size, 1, 100)

  data = data.reshape(num_images, image_size, image_size, 1)

#  print ("final data\n" , data)
  return data

def extract_images_from_test_dir(directory, image_size):

  # Get filenames from test directory
  image = []
  image_num = 0
  filenames = []
  for fnames in os.listdir(directory):  
    match = re.search(r'.csv',fnames)
    if( match ):
      print("Exact ", image_size, " entries from", fnames, "at", directory)
      
      array = load_csv_with_header(directory + fnames, 'Date', image_size, 1440)
      max_value = np.amax(array)
      my_array = np.array(array)
      if( max_value != 0 ):
        new_array = np.ceil(my_array / max_value * (image_size-2))
      else:
        new_array = np.trunc(my_array)
      image0 = draw_line_in_graph(new_array, image_size, image_size, 100, 200)
      image = np.append(image, image0)
      image_num = image_num + 1
      filenames.append(fnames)

  # Put array into image format
  image = image.reshape(image_num, image_size, image_size, 1)

  print (image_num, "test images, shape->" , image.shape)
  return image, filenames


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    np.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]


def read_data_sets(trainfile,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=False,
                   validation_size=10,
                   seed=None,
                   image_size=288, 
                   output_size=10,
                   test_dir = './data/test/',
                   train_dir = './data/'):
  if fake_data:

    def fake():
      return DataSet(
          [], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

    train = fake()
    validation = fake()
    test = fake()
    return base.Datasets(train=train, validation=validation, test=test)

  # Load train images
  if ( train_dir != None):
    train_image1 = np.load('gi-imperil.npy')

    image, image_normal_num, image_outage_num, image_plateau_num = get_train_images(train_dir, image_size, 0)
    image1, image_normal_num, image_outage_num, image_plateau_num = get_train_images(train_dir, image_size, 1)
    image2, image_normal_num, image_outage_num, image_plateau_num = get_train_images(train_dir, image_size, 2)
    image3, image_normal_num, image_outage_num, image_plateau_num = get_train_images(train_dir, image_size, 3)
    image4, image_normal_num, image_outage_num, image_plateau_num = get_train_images(train_dir, image_size, 4)
    image = np.append(image, image1)
    image = np.append(image, image2)
    image = np.append(image, image3)
    image = np.append(image, image4)

    # Set train labels
    labels_list=[]
    pattern = [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,0,0,0,0,1,1,1,1,1]
    for i in range(0, int(validation_size/25) ):
      labels_list.extend(pattern)
    for i in range(0, 5):
      labels_list.extend( [0]*image_normal_num)
      labels_list.extend( [1]*image_outage_num)
      labels_list.extend( [2]*image_plateau_num)
    
    print ("Train lable list:" , labels_list)
    image =np.concatenate((train_image1[:image_size*image_size*validation_size], image))
  elif( trainfile == None ):
    train_image1 = np.load('gi-imperil.npy')
    loaded = np.load('initialdata.npz')
    image =np.concatenate((train_image1,loaded['arr_0']))
    # Set train labels
    labels_list=[]
    pattern = [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,0,0,0,0,1,1,1,1,1]
    for i in range(0, int(image.shape[0]/image_size/image_size/25) ):
      labels_list.extend(pattern)
  else:
    train_image1 = np.load('gi-imperil.npy')
    if ( trainfile != 'no_train' ):
      train_image2 = np.load(trainfile)
      if ( trainfile[-4:] == '.npz' ):
        image =np.concatenate((train_image1,train_image2['arr_0']))
      else:        
        image =np.concatenate((train_image1[:image_size*image_size*validation_size], train_image2))
    else:
      image = train_image1[:image_size*image_size*validation_size]
    # Set train labels
    labels_list=[]
    pattern = [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,0,0,0,0,1,1,1,1,1]
    for i in range(0, int(image.shape[0]/image_size/image_size/25) ):
      labels_list.extend(pattern)
#  print ("Verfication and train data array shape:" , image.shape, ", output size:", output_size)
  train_images = image.reshape(int(image.shape[0]/image_size/image_size), image_size, image_size, 1)

  # Set the label array
  labels = np.array(labels_list)
  if ( labels.shape[0] % 100 != 0 ):
    print("Num of training images(minus validation size) has to be units of 100, Num of training images:", labels.shape[0]-validation_size)
    exit(1)
  print("Train label shape:", labels.shape)
  train_labels = dense_to_one_hot(labels, output_size)

  # Load test images
  test_images, test_files =  extract_images_from_test_dir(test_dir, image_size)
  labels = np.zeros(len(test_files), dtype=np.uint8)
  print("number of test", test_images.shape[0], labels )
  test_labels = dense_to_one_hot(labels, output_size)

  if not 0 <= validation_size <= len(train_images):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), validation_size))

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]      
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]

  options = dict(dtype=dtype, reshape=reshape, seed=seed)

  print("Train image, label shape, reshape:", train_images.shape, train_labels.shape, reshape)
  train = DataSet(train_images, train_labels, **options)
  validation = DataSet(validation_images, validation_labels, **options)
  test = DataSet(test_images, test_labels, **options)
  
  return base.Datasets(train=train, validation=validation, test=test), train_images.shape[0], test_files
