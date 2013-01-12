import PIL.Image
import numpy
from utils import tile_raster_images
import pickle as pkl

data = numpy.load("dae_mnist_weights.npy")

image = PIL.Image.fromarray(tile_raster_images(
    X=data.T,
    img_shape=(28, 28), tile_shape=(12, 12),
    tile_spacing=(1, 1)))
image.save('mnist_filters_dae.png')
