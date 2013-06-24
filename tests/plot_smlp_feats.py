import PIL.Image
import numpy
from utils import tile_raster_images
import pickle as pkl

#data = pkl.load(open("./online_smlp_lbfgs_std_adadelta.pkl"))

data = numpy.load("weights.npy")

import ipdb; ipdb.set_trace()

image = PIL.Image.fromarray(tile_raster_images(
    X=data.T,
    img_shape=(8, 8), tile_shape=(32, 64),
    tile_spacing=(1, 1)))

image.save('pento_filters_smlp.png')
