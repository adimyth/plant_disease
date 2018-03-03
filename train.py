from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import os
import h5py
import glob
import numpy as np

img_height, img_width = (100,100)
num_classes = 10
epochs = 50
batch_size = 32
best_model = './model_stuff/best_model.h5'
final_model = './model_stuff/final_model.h5'
filepath = './model/training_log.csv'
base_directory = os.path.join()

X = []
y = []
all_img_paths = glob.glob(os.path.join(base_directory, "*/*.jpg"))
np.random.shuffle(all_img_paths)
for img_path in all_img_paths:
    img = load_img(img_path)
    img = img_to_array(img)
    X.append(img)
    label = 
    y.append(label)

X_train, y_train, X_test, y_test = train_test_split(X, y, test_split=0.2)

model = Sequential()
model.add(Convolution2D(64, (3,3), activation='relu', padding='same', input_shape=(img_height, img_width,3)))
model.add(Convolution2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(128, (3,3), activation='relu', padding='same'))
model.add(Convolution2D(128, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(256, (3,3), activation='relu', padding='same'))
model.add(Convolution2D(256, (3,3), activation='relu', padding='same'))
model.add(Convolution2D(256, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(512, (3,3), activation='relu', padding='same'))
model.add(Convolution2D(512, (3,3), activation='relu', padding='same'))
model.add(Convolution2D(512, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(512, (3,3), activation='relu', padding='same'))
model.add(Convolution2D(512, (3,3), activation='relu', padding='same'))
model.add(Convolution2D(512, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

adam = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
checkpoint = ModelCheckpoint(best_model, monitor='val_loss', verbose=1, save_best_only=True)
csvlogger = CSVLogger(filepath, separator='\t')
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

from bokeh.plotting import output_file, show, figure
from bokeh.layouts import gridplot
output_file("training_plots.html")

acc = history.history['acc']
loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
x_axis = history.epoch

p_acc = figure(
    tools="pan, box_zoom, wheel_zoom, save, reset",
    x_axis_label = "Epochs",
    y_axis_label = "Training Accuracy",
    title = "Epochs vs Training Accuracy Plot",
    plot_height=500,
    plot_width=500 
)
p_acc.line(x_axis, acc, legend='training accuracy', line_color='green')

p_loss = figure(
    tools="pan, box_zoom, wheel_zoom, save, reset",
    x_axis_label = "Epochs",
    y_axis_label = "Training Loss",
    title = "Epochs vs Training Loss Plot",
    plot_height=500,
    plot_width=500 
)
p_loss.line(x_axis, acc, legend='training loss', line_color='red')

p_val_acc = figure(
    tools="pan, box_zoom, wheel_zoom, save, reset",
    x_axis_label = "Epochs",
    y_axis_label = "Validation Accuracy",
    title = "Epochs vs Validation Accuracy Plot",
    plot_height=500,
    plot_width=500 
)
p_val_acc.line(x_axis, acc, legend='validation accuracy', line_color='green')

p_val_loss = figure(
    tools="pan, box_zoom, wheel_zoom, save, reset",
    x_axis_label = "Epochs",
    y_axis_label = "Validation Loss",
    title = "Epochs vs Validation Loss Plot",
    plot_height=500,
    plot_width=500 
)
p_acc.line(x_axis, acc, legend='validation loss', line_color='red')

grid = gridplot([[p_acc, p_loss], [p_val_acc, p_val_loss]])
show(grid)