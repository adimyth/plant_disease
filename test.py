# from bokeh.io import show, output_file
# from bokeh.models import ColumnDataSource
# from bokeh.palettes import Spectral6
# from bokeh.plotting import figure
# classes = ['bacterial_spot', 'leaf_mold', 'leaf_blight', 'powdery_mildew', 'early_blight', 'yellow_leaf_curl_virus', 'spider_mites', 'mosaic_virus', 'target_spot', 'septoria_leaf_spot']
# count = [3294, 952, 2061, 2887, 2000, 5357, 1676, 373, 1404, 1771]
# source = ColumnDataSource(data=dict(classes=classes, count=count, color=Spectral6))
# p = figure(x_range=classes, y_range=(0,6000), plot_height=350, title="Sample Counts")
# p.vbar(x='classes', top='count', width=0.9, color='color', source=source)
# p.xgrid.grid_line_color = None
# # p.legend.orientation = "horizontal"
# p.xaxis.major_label_orientation = "vertical"
# show(p)


from keras.models import load_model
from keras.utils import plot_model

model = load_model("models/plant_disease.model")
plot_model(model, to_file="model.png")