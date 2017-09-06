<<<<<<< HEAD
import sys
sys.path.append('.chaper5_0shimax/src/')
=======
>>>>>>> 23dd86731ea147188fd490755d5fb2872295a946
import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
import numpy as np
import math
from evaluate import Evaluator


class Fire(chainer.Chain):
    def __init__(self, in_size, s1, e1, e3):
        super().__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(in_size, s1, 1)
            self.conv2=L.Convolution2D(s1, e1, 1)
            self.conv3=L.Convolution2D(s1, e3, 3, pad=1)

    def __call__(self, x):
        h = F.elu(self.conv1(x))
        h_1 = self.conv2(h)
        h_3 = self.conv3(h)
        h_out = F.concat([h_1, h_3], axis=1)
        return F.elu(h_out)


class FireDilated(chainer.Chain):
    def __init__(self, in_size, s1, e1, e3):
        super().__init__()
        with self.init_scope():
            self.conv1=L.DilatedConvolution2D(in_size, s1, 1)
            self.conv2=L.DilatedConvolution2D(s1, e1, 1)
            self.conv3=L.DilatedConvolution2D(s1, e3, 3, pad=2, dilate=2)

    def __call__(self, x):
        h = F.elu(self.conv1(x))
        h_1 = self.conv2(h)
        h_3 = self.conv3(h)
        h_out = F.concat([h_1, h_3], axis=1)
        return F.elu(h_out)


class FCN(chainer.Chain):
    def __init__(self, n_class, in_ch):
        super().__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(in_ch, 96, 7, stride=2, pad=3)
            self.fire2=Fire(96, 16, 64, 64)
            self.fire3=Fire(128, 16, 64, 64)
            self.fire4=Fire(128, 16, 128, 128)
            self.fire5=Fire(256, 32, 128, 128)
            self.fire6=Fire(256, 48, 192, 192)
            self.fire7=Fire(384, 48, 192, 192)
            self.fire8=Fire(384, 64, 256, 256)
            self.fire9=FireDilated(512, 64, 256, 256)

            self.score_pool1=L.Convolution2D(96, n_class, 1, stride=1, pad=0)
            self.score_pool4=L.Convolution2D(256, n_class, 1, stride=1, pad=0)
            self.score_pool9=L.Convolution2D(512, n_class, 1, stride=1, pad=0)

            self.add_layer=L.Convolution2D(n_class*3, n_class, 1, stride=1, pad=0)

            # padding means reduce pixels in deconvolution.
            self.upsample_pool4=L.Deconvolution2D(n_class, n_class, ksize= 4, stride=2, pad=1)
            self.upsample_pool9=L.Deconvolution2D(n_class, n_class, ksize= 4, stride=2, pad=1)
            self.upsample_final=L.Deconvolution2D(n_class, n_class, ksize=16, stride=4, pad=6)

        self.n_class = n_class
        self.active_learn = False
        self.evaluator = Evaluator(False, n_class)

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        h = F.elu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2)
        p1 = self.score_pool1(h)

        h = self.fire2(h)
        h = self.fire3(h)
        h = self.fire4(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        u4 = self.upsample_pool4(self.score_pool4(h))

        h = self.fire5(h)
        h = self.fire6(h)
        h = self.fire7(h)
        h = self.fire8(h)

        # h = F.max_pooling_2d(h, 3, stride=2)
        h = self.fire9(h)
        u9 = self.upsample_pool9(self.score_pool9(h))

        h = F.concat((p1, u4, u9), axis=1)
        h = self.add_layer(h)
        h = self.upsample_final(h)

        self.h = h
        self.loss = F.softmax_cross_entropy(h, t)

        self.evaluator.preparation(h, t)
        self.accuracy = self.evaluator.get_accuracy()
        self.iou = self.evaluator.get_iou()

        return self.loss
