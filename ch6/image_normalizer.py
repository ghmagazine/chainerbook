import numpy as np
from math import sqrt


class ImageNormalizer(object):
    def __normalize(self, one_channel):
        mean = np.mean(one_channel)
        var = np.var(one_channel)
        return (one_channel-mean)/float(sqrt(var))

    # global_contrast_normalization
    def GCN(self, image, args=None):
        h, w, ch = image.shape
        for i_ch in range(ch):
            image[..., i_ch] = self.__normalize(image[..., i_ch])
        return image
