import numpy
import pylab

def plot_samples(samples, no_of_rows, no_of_cols, pno=1, n_samples=30, img_shp=(28, 28)):
    if pno >= n_samples:
        return 0
    else:
        pylab.axis("off")
        pylab.subplot(no_of_rows, no_of_cols, pno)
        pylab.imshow(samples[pno].reshape(img_shp))
        print pno
        plot_samples(samples, no_of_rows, no_of_cols, pno+1, n_samples, img_shp)

if __name__=="__main__":
    data = numpy.load("data_samples.npy")
    pylab.gray()
    plot_samples(data, 3, 10, pno=0)
    pylab.show()
