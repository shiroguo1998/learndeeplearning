# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

def parse_csv(line):
  example_defaults = [[0.], [0.], [0.], [0.], [0]]  # sets field types
  parsed_line = tf.decode_csv(line, example_defaults)

  # First 4 fields are features, combine into single tensor
  features = tf.reshape(parsed_line[:-1], shape=(4,))
  # Last field is the label
  label = tf.reshape(parsed_line[-1], shape=())
  return features, label
    
def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels = y, logits = y_) # calculate the loss

def grad(model, inputs, targets):
    with tfe.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)

# set path to save download file
PATH = "/home/shiro/Project/DeepLearning/Iris_classification/"

# enable tf to execute values directly instead of creating graphh
tf.enable_eager_execution()

# some initial tests
print("Tensorflow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

#url to the download file
train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"

#get file and save to path
train_dataset_fp = tf.keras.utils.get_file(
        fname=PATH + os.path.basename(train_dataset_url),
        origin=train_dataset_url)

train_dataset = tf.data.TextLineDataset(train_dataset_fp) # read the file
train_dataset = train_dataset.skip(1).map(parse_csv) # skip the header row and map all row to parse_csv to parse
train_dataset = train_dataset.shuffle(buffer_size=1000)  # randomize the dataset
train_dataset = train_dataset.batch(32) # combine data to batches of 32 rows

# view 1 example from a batch in the dataset
features, label = tfe.Iterator(train_dataset).next()
print("Example feature: ", features[0])
print("Example label: ", label[0])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation = "relu", input_shape = (4,)),
    tf.keras.layers.Dense(10, activation = "relu"),
    tf.keras.layers.Dense(3),
])

optimizer = tf.train.AdamOptimizer()

train_loss_result = []
train_accuracy_result = []

num_epochs = 201 # number of iterations

for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()
    
    #train loop on batch of 32
    for (x,y) in tfe.Iterator(train_dataset):
        grads = grad(model, x, y)
        #optimize model
        optimizer.apply_gradients(zip(grads, model.variables), 
                                  global_step = tf.train.get_or_create_global_step())
        #track progress
        epoch_loss_avg(loss(model, x, y))
        epoch_accuracy(tf.argmax(model(x), axis = 1, output_type = tf.int32), y)
        
    train_loss_result.append(epoch_loss_avg.result())
    train_accuracy_result.append(epoch_accuracy.result())
    
    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))
        
fig, axes = plt.subplots(2, sharex = True, figsize = (12, 8))
fig.suptitle("Training metrics")

axes[0].set_ylabel("Loss", fontsize = 14)
axes[0].plot(train_loss_result)

axes[1].set_ylabel("Accuracy", fontsize = 14)
axes[1].set_xlabel("Epoch", fontsize = 14)
axes[1].plot(train_accuracy_result)

plt.show()