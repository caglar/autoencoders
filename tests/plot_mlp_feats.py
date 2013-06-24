import PIL.Image
import numpy
from utils import tile_raster_images
import pickle as pkl

data = numpy.load("/u/gulcehrc/Documents/Papers/lisa_lab/articles/2012/intermediate_targets/prmlp_1stlayer.npy")

image = PIL.Image.fromarray(tile_raster_images(
    X=data.T,
    img_shape=(8, 8), tile_shape=(32, 64),
    tile_spacing=(1, 1)))
image.save('pento_filters_ikgnn.png')
