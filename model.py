import os
import csv
import cv2
import numpy as np
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Flatten, Dense, Lambda, Convolution2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint


# Initialize parameters for the whole program
correction = 0.2
lr = 0.001
beta_1 = 0.9
beta_2 = 0.999
epsilon = None
decay = 0.0
rate = 0.5


# Open the driving_log.csv file which contains the images names captured by three different
# camera (center, left, right).
lines = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Split the training data and testing data with ratio 8:2
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# Define the generator as it creates batch and output them
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for line in batch_samples:
                # Get paths of three types of images (center, left, right)
                center_path = line[0]
                left_path = line[1]
                right_path = line[2]
                # Extract the filename of them
                center_filename = center_path.split('/')[-1]
                left_filename = left_path.split('/')[-1]
                right_filename = right_path.split('/')[-1]
                current_center_path = 'IMG/' + center_filename
                current_left_path = 'IMG/' + left_filename
                current_right_path = 'IMG/' + right_filename
                # Read all images and store them
                center_image = cv2.imread(current_center_path)
                left_image = cv2.imread(current_left_path)
                right_image = cv2.imread(current_right_path)
                # Convert BGR into RGB images
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                # Append them into images, first center_image, then left_image and last right_image
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                # Append the angles into measurements, first center_angle, then left_angle and last right_angle
                center_measurement = float(line[3])
                left_measurement = center_measurement + correction
                right_measurement = center_measurement - correction
                measurements.append(center_measurement)
                measurements.append(left_measurement)
                measurements.append(right_measurement)
            # Store the flipped images and its steering_angle
            augmented_images = []
            augmented_measurements = []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement * -1.0)

            # Setting X data and y data, where x data contains images from different angle, and y data contains its
            # corresponding angles.
            x = np.array(augmented_images)
            y = np.array(augmented_measurements)
            # Shuffle the data
            yield shuffle(x, y)


# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# Network Architecture
model = Sequential()
# Normalizing the data into range [-1, 1].
model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160, 320, 3)))
# Cropping the image and keep the area where we are interested.
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
# Input shape (90, 320, 3), output shape (43, 158, 24) Valid padding
model.add(Convolution2D(filters=24, kernel_size=5, strides=2, padding='valid', activation='relu'))
# Input shape (43, 158, 24), output shape (20, 77, 36) valid padding
model.add(Convolution2D(filters=36, kernel_size=5, strides=2, padding='valid', activation='relu'))
# Input shape (20, 77, 36), output shape (8, 37, 48) valid padding
model.add(Convolution2D(filters=48, kernel_size=5, strides=2, padding='valid', activation='relu'))
# Input shape (8, 37, 48), output shape (6, 35, 64) valid padding
model.add(Convolution2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu'))
# Input shape (6, 35, 64), output shape (4, 33, 64) valid padding
model.add(Convolution2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Setting up the optimizer as Adam optimizer with lr = 0.001, beta_1 = 0.9, beta_2 = 0.999 and no weight_decay.
sgd = optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay, amsgrad=False)
model.compile(loss='mse', optimizer=sgd)

# Storing the weight into file "model.hdf5"
file_path = "model.hdf5"

# Setting up a check pointer to monitor the validation loss and pick the best weights among them.
check_pointer = ModelCheckpoint(filepath=file_path, monitor='val_loss', verbose=1, save_best_only=True)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5, callbacks=[check_pointer])

