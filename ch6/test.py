import os, sys, time
from mini_batch_loader import DatasetPreProcessor
from fcn_squeeze_dilate import FCN
import settings  # 設定の読み込み

import chainer
import chainer.functions as F
from chainer import serializers
from chainer import cuda, optimizers, Variable

import numpy as np
from skimage import io
import pandas as pd
from matplotlib import pyplot as plt


def test(model):
    # データセットのロード
    test_mini_batch_loader = \
        DatasetPreProcessor(chainer.global_config.user_test_args)
    test_data_size = test_mini_batch_loader.__len__()

    print("------test data size")
    print(test_data_size)

    sum_accuracy = 0
    sum_iou      = 0
    for idx in range(test_data_size):
        raw_x, raw_t = test_mini_batch_loader.load_data([idx])

        x = chainer.Variable(raw_x)
        t = chainer.Variable(raw_t)

        model(x, t)

        print("accuracy:", model.accuracy)
        sum_accuracy += model.accuracy
        sum_iou      += np.array(model.iou)

        # 出力ラベルを画像に変換
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
            .format(
                sum_iou/test_data_size,
                np.sum(sum_iou/test_data_size)/chainer.global_config.user_test_args.n_class))


def restoration(result_labels, color_data, output_img_path, i_batch, n_class):
    for i_image, labels in enumerate(result_labels):
        print("---------------labels")
        print(labels.shape)
        h, w = labels.shape
        # 出力ラベルから画像への変換
        img = np.zeros((h, w, 3))
        for category in range(n_class):
            idx = np.where(labels==category)  # indexはタプルに格納される
            if len(idx[0])>0:
                color = color_data.ix[category]
                img[idx[0], idx[1], :] =[color['r'], color['g'], color['b']]
            img = img.astype(np.uint8)
        io.imsave(output_img_path+'/test_result_{}.jpg' \
                                    .format(str(i_batch).zfill(5)), img)
        # plt.figure()
        # io.imshow(img)
        # plt.show()

def main():
    # モデルのロード
    model = FCN(chainer.global_config.user_test_args.n_class,
                chainer.global_config.user_test_args.in_ch)
    if os.path.exists(
        chainer.global_config.user_test_args.model_path.format(
            chainer.global_config.user_test_args.model_version)):
        serializers.load_npz(
            chainer.global_config.user_test_args.model_path.format(
                chainer.global_config.user_test_args.model_version), model)

    # テストの実行
    test(model)


if __name__ == '__main__':
    chainer.global_config.train = chainer.global_config.user_test_args.train
    start = time.time()
    main()
    end = time.time()
    print("{}[m]".format((end - start)/60))
