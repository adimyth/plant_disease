from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import os

output_path = 'data/spotted_wilt_virus/{}.jpg'
count = 10

gen = ImageDataGenerator(
    rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=1.5,
    horizontal_flip=True,
    vertical_flip=True
)

for j in range(50):
    if os.path.exists('data/Wilt-3/{}.jpg'.format(j)):
        input_path = 'data/Wilt-3/{}.jpg'.format(j)
        # load image to array
        image = img_to_array(load_img(input_path))

        # reshape to array rank 4
        image = image.reshape((1,) + image.shape)

        k = j*10+1008

        # let's create infinite flow of images
        images_flow = gen.flow(image, batch_size=1)
        for i, new_images in enumerate(images_flow):
            # we access only first image because of batch_size=1
            new_image = array_to_img(new_images[0], scale=True)
            new_image.save(output_path.format(i + 1 + k))
            if i >= count:
                break