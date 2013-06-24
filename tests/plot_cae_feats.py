import PIL.Image
import numpy
from utils import tile_raster_images
import pickle as pkl

data = numpy.load("cae_mnist_weights_l1.npy")

image = PIL.Image.fromarray(tile_raster_images(
    X=data.T,
    img_shape=(28, 28), tile_shape=(30, 30),
    tile_spacing=(1, 1)))
image.save('mnist_filters_cae.png')
