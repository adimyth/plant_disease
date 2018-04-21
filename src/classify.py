# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2
import os
import matplotlib.pyplot as plt

image_path = os.path.join("..", "sample.png")
model_path = os.path.join("..", "models" , "plant_disease.model")
binarizer_path = os.path.join("..", "models" , "plant_disease.pickle")

# load the image
image = cv2.imread(image_path)
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
print(lb.classes_)

# classify the input image
print("[INFO] classifying image...")
proba = model.predict(image)[0]
idx = np.argmax(proba)
label = lb.classes_[idx]
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)
cv2.putText(output, str(np.max(proba)), (10, 55),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 0, 255), 2)

# # show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()