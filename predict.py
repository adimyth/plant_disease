from keras.models import load_model
import os
import h5py
import cv2
import numpy as np
import itertools
from keras.preprocessing.image import load_img, img_to_array
from bokeh.plotting import Figure, show, output_file
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot
from bokeh.palettes import Category10 as palette

labels = ['BacterialSpot', 'EarlyBlight', 'LeafBlight', 'LeafMold', 'MosaicVirus', 'PowderyMildew', 'SeptoriaLeafSpot', 'SpiderMites', 'TargetSpot', 'YellowLeafCurlVirus']
model_path = os.path.join("model_stuff", "trained_model.h5")
img_width, img_height = (70, 70)

def prediction(image):
    # loading model
    model = load_model(model_path)

    # reading image
    img = cv2.imread(image)
    img = cv2.resize(img, (70,70))
    # img /= 255

    # processing image
    blurImg = cv2.GaussianBlur(img, (5, 5), 0)   
    hsvImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)  
    lower_green = (25, 40, 50)
    upper_green = (75, 255, 255)
    mask = cv2.inRange(hsvImg, lower_green, upper_green)  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    bMask = mask > 0  
    clear = np.zeros_like(img, np.uint8)
    clear[bMask] = img[bMask]

    # cv2.imwrite("processed.png", clear)
    # clear /= 255
    clear = np.expand_dims(clear, axis=0)

    # making prediction
    prediction = model.predict(clear)
    print(prediction)
    x = labels
    y = list(prediction)

    output_file("plots.html")
    colors = itertools.cycle(palette)

    # displaying the image    
    # url = str(image)
    # source = ColumnDataSource(data=dict(url = url))
    # p1 = Figure(x_range=(0,1), y_range=(0,1), plot_width=600, plot_height=400)
    # p1.image_url(url='url', x=0, y=1, h=1, w=1, source=source)
    
    # plotting the barplot
    source = ColumnDataSource(data=dict(x=x, y=y, color=colors))
    p2 = Figure(x_range=labels, y_range=(0,1), plot_height=400, plot_width=600, title="Class Probabilites")
    p2.xaxis.major_label_orientation = "vertical"
    p2.vbar(x='x', top='y',width=0.9, color='color', legend='x', source=source)

    # layout 
    grid = gridplot([[p2]])
    show(grid)