from __future__ import print_function

import matplotlib
matplotlib.use("Qt4Agg")
from matplotlib import pyplot as plt

import collections
import os
import math
import numpy
import chainer
from random import shuffle
from chainer.cuda import cupy
import itertools
from past.builtins import xrange


from matplotlib.backends.backend_pdf import PdfPages

from PIL import Image, ImageOps
#from skimage import color

class FlexibleImageDataset(chainer.datasets.LabeledImageDataset):

    def __init__(self, paths, root='.', mean=0.0, scale=1.0, size=(128,128), dtype=numpy.float32,
                 label_dtype=numpy.int32, withlabels=True):

        super(FlexibleImageDataset, self).__init__(paths, root, dtype, label_dtype)
        self.mean = numpy.asarray(mean)
        self.size = size
        self.withlabels = withlabels
        self.scale = scale
        self.num_classes = None
        self.calc_num_classes()

    def calc_num_classes(self):
        classes = [x[1] for x in self._pairs]
        self.num_classes = len(set(classes))

    def get_num_classes(self):
        return self.num_classes

    def get_class_weights(self, gpu_id=-1):
        classes = [x[1] for x in self._pairs]
        collect = collections.Counter(classes)
        weights = [float(item[1]) for item in collect.iteritems()]
        total = sum(weights)
        weights[:] = [ 1.0 - (x / total) for x in weights]
        if gpu_id >= 0:
            return cupy.array(weights, "float32")
        else:
            return numpy.array(weights, "float32")

    def summary(self, freq=False):
        print("Dataset contains {} entries".format(len(self._pairs)))
        classes = [x[1] for x in self._pairs]
        print("Number of classes: {}".format(len(set(classes))))
        if freq:
            print(collections.OrderedDict(collections.Counter(classes)))

    def make_pdf(self, filename="dataset.pdf"):
        with PdfPages(filename) as pdf:
            if self.num_classes is None:
                self.calc_num_classes()

            for cls in range(self.num_classes):
                print("class {} of {}".format(cls, self.num_classes))
                samples = [x[0] for x in self._pairs if int(x[1]) is int(cls)]
                print("class {} contains {} samples".format(cls, len(samples)))
                if not samples:
                    continue
                plt.figure()
                plt.axis('off')

                isize = (64,64)#(32,32)
                imagesize = (1024,1024)
                grid = Image.new("RGB", imagesize, color=(255,255,255))
                tmplt_path = "/mnt/sakuradata2/calland/software/traffic-sign-data-synthesizer/templates/GTSRB/original/{}.jpg".format(cls)
                template = Image.open(tmplt_path)
                template = template.resize(isize, Image.ANTIALIAS)
                grid.paste(template, (0,0))

                maximg = math.floor((imagesize[0] / isize[0]) * (imagesize[1] / isize[1]))

                index = 0   
                for i, j in itertools.product(xrange(isize[0],imagesize[0],isize[0]), xrange(1,imagesize[1],isize[1])):
                    if index == len(samples) or index >= maximg:
                        break
                    im = Image.open(samples[index])
                    im = im.resize(isize, Image.ANTIALIAS)
                    grid.paste(im, (j,i))
                    index += 1

                plt.imshow(grid, interpolation="nearest")

                pdf.savefig(dpi=96)  # saves the current figure into a pdf page
                plt.close()

    def get_example(self, i, transpose=True):
        path, int_label = self._pairs[i]
        full_path = os.path.join(self._root, path)

        f = Image.open(full_path).convert("RGB")
        f = f.resize(self.size)#, Image.ANTIALIAS)

        try:
            image = numpy.asarray(f, dtype=self._dtype)
        finally:
            # Only pillow >= 3.0 has 'close' method
            if hasattr(f, 'close'):
                f.close()

        #if image.ndim == 2:
            # image is greyscale
        #    image = image[:, :, numpy.newaxis]
        
        label = numpy.array(int_label, dtype=self._label_dtype)

        #print(image)
        image /= 255.0
        #image = color.rgb2lab(image)
        #print(image)
        image *= self.scale
        #print(image)
        image -= self.mean

        image = image.astype(self._dtype)
        #print("mean {} std {}".format(image.mean(), image.std()))
        #exit()
        if transpose:
            image = image.transpose(2,0,1)

        if self.withlabels:
            return image, label
        else:
            return image
