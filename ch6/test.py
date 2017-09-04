import os, sys, time
from mini_batch_loader import DatasetPreProcessor
from fcn_squeeze_dilate import FCN
import settings  # import get_args

import chainer
import chainer.functions as F
from chainer import serializers
from chainer import cuda, optimizers, Variable

import numpy as np
from skimage import io
import pandas as pd
from matplotlib import pyplot as plt


def test(model):
    # load dataset
    test_mini_batch_loader = \
        DatasetPreProcessor(chainer.global_config.user_test_args)
    test_data_size = test_mini_batch_loader.__len__()

    print("------test data size")
    print(test_data_size)

    sum_accuracy = 0
    sum_iou      = 0
    for idx in range(test_data_size):
        raw_x, raw_t = test_mini_batch_loader.load_data([idx])

        x = chainer.Variable(raw_x, volatile=True)
        t = chainer.Variable(raw_t, volatile=True)

        model(x, t)

        print("accuracy:", model.accuracy)
        sum_accuracy += model.accuracy
        sum_iou      += np.array(model.iou)

        # image restoration
        result_labels = model.h.data.argmax(axis=1)
        color_data = pd.read_csv(chainer.global_config.user_test_args.label_path)
        restoration(
            result_labels,
            color_data,
            chainer.global_config.user_test_args.output_path,
            idx,
            chainer.global_config.user_test_args.n_class)

    print("test mean accuracy {}".format(sum_accuracy/test_data_size))
    print("test mean IoU {}, meanIU {}" \
            .format(sum_iou/test_data_size, np.sum(sum_iou/test_data_size)/n_class))


def restoration(result_labels, color_data, output_img_path, i_batch, n_class):
    # input images
    for i_image, labels in enumerate(result_labels):
        print("---------------labels")
        print(labels.shape)
        h, w = labels.shape
        # label outputed
        img = np.zeros((h, w, 3))
        for category in range(n_class):
            idx = np.where(labels==category)  # index is a tuple object
            if len(idx[0])>0:
                color = color_data.ix[category]
                img[idx[0], idx[1], :] =[color['R'], color['G'], color['B']]
            img = img.astype(np.uint8)
        io.imsave(output_img_path+'/test_result_{}.jpg' \
                                    .format(str(i_batch).zfill(5)), img)
        plt.figure()
        io.imshow(img)
        plt.show()

def main():
    # load FCN model
    model = FCN(chainer.global_config.user_test_args.n_class,
                chainer.global_config.user_test_args.in_ch)
    if os.path.exists(
        chainer.global_config.user_test_args.model_path.format(
            chainer.global_config.user_test_args.model_version)):
        serializers.load_npz(
            chainer.global_config.user_test_args.model_path.format(
                chainer.global_config.user_test_args.model_version), model)

    # testing
    test(model)


if __name__ == '__main__':
    chainer.global_config.train = chainer.global_config.user_test_args.train
    start = time.time()
    main()
    end = time.time()
    print("{}[m]".format((end - start)/60))
