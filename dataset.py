import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio
import os

def preprocess_image(filename):

    image_string = tf.io.read_file(filename)
    # Decodes a .tiff encoded image tensor of RGBA format (Output shape-[height,width,4])
    image = tfio.experimental.image.decode_tiff(image_string)
    # Selecting only the 3 channels of the image - RGB
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    # Combines multiple tensors
    image = tf.stack([r,g,b],axis=-1)
    # Resizing the image
    image = tf.image.resize(image, (180,180))
    return image

def preprocess_triplets(anchor,positive,negative):
  return [
      preprocess_image(anchor),
      preprocess_image(positive),
      preprocess_image(negative),
  ]

# Here, First 30 signatures in each folder are forged Next 24 signatures are genuine
orig_groups, forg_groups = [], []
for directory in dir_list:
    images = os.listdir(directory)
    images.sort()
    images = [directory+'/'+x for x in images]
    forg_groups.append(images[:30])
    orig_groups.append(images[30:])


anchor_images = []
positive_images = []
negative_images = []

# Consists of all forgery images
forg_groups_all = tf.reduce(lambda x,y:x+y,forg_groups)

for i in range(len(orig_groups)):
    orig_gp = orig_groups[i]
    forg_gp = forg_groups[i]
    for j in range(len(orig_gp)):
      for k in range(j+1,len(orig_gp)):
        anchor = orig_gp[j]
        positive = orig_gp[k]
        negative = forg_groups_all[np.random.randint(len(forg_groups_all))]
        anchor_images.append(anchor)
        positive_images.append(positive)
        negative_images.append(negative)


anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images[:20000])
positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images[:20000])
negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images[:20000])

dataset = tf.data.Dataset.zip((anchor_dataset,positive_dataset,negative_dataset))



dataset = dataset.shuffle(buffer_size=len(dataset))
dataset = dataset.map(preprocess_triplets)

# Training Partition
train_data = dataset.take(round(len(dataset)*0.8))
train_data = train_data.batch(8)
train_data = train_data.prefetch(4)

val_data = dataset.skip(round(len(dataset)*0.8))
val_data = val_data.take(round(len(dataset)*0.15))
val_data = val_data.batch(8)
val_data = val_data.prefetch(4)

