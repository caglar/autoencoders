import PIL.Image
import numpy
from utils import tile_raster_images
import pickle as pkl

data = numpy.load("pae_mnist_enc_weights.npy")

image = PIL.Image.fromarray(tile_raster_images(
    X=data.T,
    img_shape=(48, 48), tile_shape=(4, 600),
    tile_spacing=(1, 1)))

image.save('filters_ae.png')
