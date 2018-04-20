from keras.models import load_model
import os
import h5py
import cv2
import numpy as np
import io
from keras.preprocessing.image import load_img, img_to_array
from bokeh.plotting import Figure, show, output_file
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot
from bokeh.palettes import Spectral6
import matplotlib.pyplot as plt

labels = []
model_path = os.path.join("model_stuff", "trained_model.h5")
img_width, img_height = (70, 70)
image = '/home/aditya/plant_disease/train/bacterial_spot/0a0dbf1f-1131-496f-b337-169ec6693e6f___NREC_B.Spot 9241.JPG'

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