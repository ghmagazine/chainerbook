import sys, time, os
import settings  # import get_args
from mini_batch_loader import DatasetPreProcessor
from fcn_squeeze_dilate import FCN

import chainer
import chainer.functions as F
from chainer import serializers
from chainer import cuda, optimizers, Variable

import numpy as np
np.random.seed(555)
from skimage import io
import math


def prepare_dataset():
    # load dataset
    train_mini_batch_loader = \
        DatasetPreProcessor(chainer.global_config.user_train_args)
    train_it = chainer.iterators.SerialIterator(
                train_mini_batch_loader,
                chainer.global_config.user_train_args.training_params.batch_size)
    return train_mini_batch_loader, train_mini_batch_loader.__len__()


#@profile
def main():
    # load dataset
    train_mini_batch_loader, train_data_size = prepare_dataset()
    # load FCN model
    model = FCN(chainer.global_config.user_train_args.n_class,
                chainer.global_config.user_train_args.in_ch)

    # setup
    # optimizer = chainer.optimizers.RMSpropGraves(lr=args.training_params.learning_rate)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(
        chainer.optimizer.WeightDecay(
            chainer.global_config.user_train_args.training_params.weight_decay))

    # training
    for epoch in range(1, 100):
        print("epoch %d" % epoch)
        sum_accuracy = 0
        sum_loss     = 0
        sum_iou      = np.zeros(chainer.global_config.user_train_args.n_class)

        all_indices = np.random.permutation(train_data_size)
        batch_size = chainer.global_config.user_train_args.training_params.batch_size
        for i in range(0, train_data_size, batch_size):
            batch_indices = all_indices[i:i+batch_size]
            raw_x, raw_t = train_mini_batch_loader.load_data(batch_indices)
            x = chainer.Variable(raw_x)
            t = chainer.Variable(raw_t)

            model.zerograds()
            loss = model(x, t)
            loss.backward()
            optimizer.update()

            if math.isnan(loss.data):
                raise RuntimeError("ERROR in main: loss.data is nan!")

            sum_loss += loss.data * batch_size
            sum_accuracy += model.accuracy * batch_size
            sum_iou += np.array(model.iou) * batch_size

        print("train mean loss {}, accuracy {}, IoU {}" \
                .format(sum_loss/train_data_size, sum_accuracy/train_data_size,
                        sum_iou/train_data_size))
        # saving
        snapshot_epochs = \
            chainer.global_config.user_train_args.training_params.snapshot_epochs
        if epoch % snapshot_epochs == 0:
            stor_dir = os.path.dirname(
                chainer.global_config.user_train_args.model_path.format(epoch))
            if not os.path.exists(stor_dir):
                os.makedirs(stor_dir)
            serializers.save_npz(args.model_path.format(epoch), model)

    # saving
    serializers.save_npz(
        chainer.global_config.user_train_args.model_path.format("final"), model)


if __name__ == '__main__':
    chainer.global_config.train = chainer.global_config.user_train_args.train
    start = time.time()
    main()
    end = time.time()
    print("{}[m]".format((end - start)/60))
