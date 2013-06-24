import numpy
import pylab
import cPickle as pkl

def colored_axis(axes, color="red"):
    axes.spines['bottom'].set_color(color)
    axes.spines['top'].set_color(color)
    axes.spines['right'].set_color(color)
    axes.spines['left'].set_color(color)
    axes.get_yaxis().set_ticks([])
    axes.get_xaxis().set_ticks([])

def plot_samples(samples, no_of_rows, no_of_cols, pno=1, n_samples=100, plot_every=1, img_shp=(28,
    28), axes=None):
    if pno == 0:
        axes = pylab.subplot(no_of_rows, no_of_cols, pno + 1, aspect='equal')
        colored_axis(axes, "red")
        pylab.imshow(samples[pno].reshape(img_shp))
        plot_samples(samples,
                no_of_rows,
                no_of_cols,
                pno=pno + plot_every,
                n_samples=n_samples,
                plot_every=plot_every,
                img_shp=img_shp,
                axes=axes)

    if pno >= n_samples:
       colored_axis(axes, "black")
       return 0
    else:
        plot_no = pno / plot_every
        axes = pylab.subplot(no_of_rows, no_of_cols, plot_no + 1, aspect='equal')
        colored_axis(axes, "black")
        pylab.imshow(samples[pno].reshape(img_shp))
        plot_samples(samples,
                no_of_rows,
                no_of_cols,
                pno=pno + plot_every,
                n_samples=n_samples,
                plot_every=plot_every,
                img_shp=img_shp,
                axes=axes)

if __name__=="__main__":
    data = numpy.load("data_samples_l2_dae.npy")
    #data = pkl.load(open("samples_file.pkl"))
    import ipdb; ipdb.set_trace()
    pylab.gray()
    plot_samples(data, 5, 8, pno=0, n_samples=40, plot_every=1, img_shp=(30, 30))
    pylab.show()
