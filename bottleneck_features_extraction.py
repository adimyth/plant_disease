import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import glob
from keras.callbacks import ModelCheckpoint, CSVLogger
import string
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils import plot_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from bokeh.plotting import output_file, show, figure
from bokeh.layouts import gridplot

# dimensions of our images.
img_width, img_height = 150, 150

top_model_path = 'bottleneck_model.h5'
train_data_dir = 'data'
filepath = 'best_model.h5'
epochs = 30
batch_size = 16

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(generator)
    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    directories = ['early_blight', 'late_blight', 'leaf_mould', 'magnesium_defeciency', 'nitrogen', 'spotted_wilt_virus', 'yellow_leaf_curl_virus']
    train_length=[]
    for i in directories:
        for j in range(0,len(glob.glob('data/'+i+'/*'))):
            train_length.append(i)
    train_labels = np.ndarray.flatten(np.array(train_length))
    
    encoder = LabelEncoder()
    train_labels = encoder.fit_transform(train_labels)
    train_labels = np_utils.to_categorical(train_labels, num_classes=26)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(26, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    plot_model(model,to_file='model.png')

    csvlogger = CSVLogger('training_log.csv')
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True)

    history = model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size, 
              callbacks=[csvlogger, checkpoint])

    output_file("training_plots.html")

    acc = history.history['acc']
    loss = history.history['loss']
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


    grid = gridplot([[p_acc, p_loss]])
    show(grid)
    model.save(top_model_path)

save_bottlebeck_features()
train_top_model()