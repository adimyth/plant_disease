# necessary imports
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2
import os
from skimage import io
import matplotlib.pyplot as plt
# bokeh imports
import bokeh
from bokeh.layouts import column
from bokeh.plotting import Figure, show, output_file, figure
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot
from bokeh.palettes import Spectral6


def prediction():
    img_path = os.path.join(os.getcwd(), 'upload_folder', os.listdir('upload_folder')[0])
    model_path = os.path.join('models', 'plant_disease.model')
    binarizer_path = os.path.join('models', 'plant_disease.pickle')

    # reading only the first image from the upload_folder
    image = cv2.imread(img_path) 
    output = image.copy()
    
    # pre-process the image for classification
    image = cv2.resize(image, (80, 80))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # load the trained convolutional neural network and the label
    # binarizer
    print("[INFO] loading network...")
    model = load_model(model_path)
    lb = pickle.loads(open(binarizer_path, "rb").read())

    # classify the input image
    print("[INFO] classifying image...")
    proba = model.predict(image)[0]
    idx = np.argmax(proba)
    label = lb.classes_[idx]
    probability = str(np.max(proba))

    # show the output image
    classes = lb.classes_
    source = ColumnDataSource(data=dict(classes=classes, probability=proba, color=Spectral6))

    plot_height = 600
    plot_width = 800
    color_mapper = None

    # barplot
    p = Figure(x_range=classes, y_range=(0,1), plot_height=plot_height, plot_width=plot_width, title="Class Probabilites")
    p.vbar(x='classes', top='probability', width=0.9, source=source)
    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation = "vertical"
    p.xaxis.major_label_text_font_style = "italic"
    p.xaxis.major_label_text_font_size = "12pt"
    p.yaxis.major_label_text_font_size = "12pt"

    # image
    p1 = figure(x_range=(0,1), y_range=(0,1))
    print(img_path)
    p1.image_url(url=[img_path], x=0, y=1, h=1, w=1)
    
    # plotting it altogether
    show(column(p1, p))
# prediction()