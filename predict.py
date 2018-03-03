from keras.models import load_model
import os
import h5py
import io
from keras.preprocessing.image import load_img, img_to_array
from bokeh.plotting import Figure, show, output_file
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot
from bokeh.palettes import Spectral6

labels = []
model_path = os.path.join("model_stuff", "model_weights.h5")
img_width, img_height = (100, 100)

def prediction(image):
    model = load_model(model_path)
    img = load_img(image, target_size=(img_height, img_width))
    img = img_to_array(img)
    prediction = model.predict_classes(img)
    x = labels
    y = list(prediction)

    output_file("plots.html")
    # displaying the image
    url = str(image)
    source = ColumnDataSource(data=dict(url = url))
    p1 = Figure(x_range=(0,1), y_range=(0,1), plot_width=600, plot_height=400)
    p1.image_url(url='url', x=0, y=1, h=1, w=1, source=source)
    
    # plotting the barplot
    source = ColumnDataSource(data=dict(x=x, y=y, color=Spectral6))
    p2 = Figure(x_range=labels, y_range=(0,1), plot_height=400, plot_width=600, title="Class Probabilites")
    p2.vbar(x='x', y='y',width=0.9, color='color', legend='x', source=source)

    # layout 
    grid = gridplot([[p1, p2]])
    show(grid)