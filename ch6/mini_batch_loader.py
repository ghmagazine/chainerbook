import os, sys
import chainer
from matplotlib import pylab as plt
import numpy as np
np.random.seed(555)
from image_normalizer import ImageNormalizer
from skimage import io, transform
import numbers
import pandas as pd


class DatasetPreProcessor(chainer.dataset.DatasetMixin):
    def __init__(self, args):
        self.args = args
        self.image_normalizer = ImageNormalizer()
        self.pairs = self.read_paths()
        self.counter = 0
        self.image_size_in_batch = [None, None]  # height, width

    def __len__(self):
        return len(self.pairs)

    def read_paths(self):
        path_label_pairs = []
        for image_path, label in self.__path_label_pair_generator():
            path_label_pairs.append((image_path, label))
        return path_label_pairs

    def __path_label_pair_generator(self):
        with open(self.args.image_pointer_path, 'r') as f_image:
            for image_file_name in f_image:
                image_file_name = image_file_name.rstrip()
                image_full_path = os.path.join(self.args.image_dir_path, image_file_name)

                label_file_name = image_file_name.replace('.png', '_L.npz')
                label_full_path = os.path.join(self.args.image_dir_path, label_file_name)
                if os.path.isfile(image_full_path) and os.path.isfile(label_full_path):
                    yield image_full_path, np.load(label_full_path)['data']
                else:
                    assert False, "error occured at path_label_pair_generator(file is not fined)."

    def __init_batch_counter(self):
        if self.args.train and self.counter==self.args.training_params.batch_size:
            self.counter = 0
            self.image_size_in_batch = [None, None]

    def __set_image_size_in_batch(self, image):
        if self.counter==1:
            resized_h, resized_w = image.shape[:2]
            self.image_size_in_batch = [resized_h, resized_w]

    def load_data(self, indices):
        xs = []
        ys = []

        for idx in indices:
            batch_inputs = self.get_example(idx)
            xs.append(batch_inputs[0])
            ys.append(batch_inputs[1])
        return np.array(xs, dtype=np.float32), np.array(ys, np.int32),

    def get_example(self, index):
        self.counter += 1
        if self.args.debug_mode:
            if self.counter>15:
                assert False, 'stop test'

        path, gt = self.pairs[index]
        image = io.imread(path)

        if image is None:
            raise RuntimeError("invalid image: {}".format(path))

        if self.args.debug_mode:
            plt.figure()
            io.imshow(image)
            plt.show()

        h, w, ch = image.shape

        image, ms, ds = self.resize_image(image)
        gt = gt[ds[0]:(gt.shape[0]-ms[0]+ds[0]),
                ds[1]:(gt.shape[1]-ms[1]+ds[1])]

        image, ms, ds = self.resize_image(image, 0.25)
        gt = gt[::4, ::4]
        gt = gt[ds[0]:(gt.shape[0]-ms[0]+ds[0]),
                ds[1]:(gt.shape[1]-ms[1]+ds[1])]

        if self.args.debug_mode:
            plt.figure()
            io.imshow(image)
            plt.show()

            color_data = pd.read_csv(self.args.label_path)
            restoration([gt], color_data, './', index, 32)

<<<<<<< HEAD
        # augmentat image
        if self.args.aug_params.do_augment:
            image, gt = self.augment_image(image, gt)

        # store image size
        # because dimension must be equeal per batch
        self.__set_image_size_in_batch(image)

        # image normalize
=======
        # augmentation
        if self.args.aug_params.do_augment:
            image, gt = self.augment_image(image, gt)

        # バッチごとにデータサイズを統一する
        self.__set_image_size_in_batch(image)

        # 画像の正規化
>>>>>>> 23dd86731ea147188fd490755d5fb2872295a946
        image = self.image_normalizer.GCN(image)

        if self.args.debug_mode:
            show_image = image.astype(np.uint8)
            plt.figure()
            io.imshow(show_image)
            plt.show()

<<<<<<< HEAD
        # transpose for chainer
        image = image.transpose(2, 0, 1)
        # initialize batch counter
=======
        # Chainerの入力に合わせてメモリオーダーを変更
        image = image.transpose(2, 0, 1)
        # バッチカウンターの初期化
>>>>>>> 23dd86731ea147188fd490755d5fb2872295a946
        self.__init_batch_counter()

        batch_inputs = image.astype(np.float32), np.array(gt, dtype=np.int32)
        return batch_inputs

    def augment_image(self, image, gt):
        orig_h, orig_w, _ = image.shape
        if self.args.aug_params.params.do_scale and self.counter==1:
            image, ms, ds = self.scaling(image)
            h, w, ch = image.shape
            inv_scale = orig_h//h, orig_w//w
            gt = gt[::inv_scale[0], ::inv_scale[1]]

        if self.args.aug_params.params.do_flip:
            image, gt = self.flip(image, gt)

        if self.args.aug_params.params.change_britghtness:
            image = self.random_brightness(image)

        if self.args.aug_params.params.change_contrast:
            image = self.random_contrast(image)

        return image, gt

    def resize_image(self, image, scale=None):
        xh, xw = image.shape[:2]

        if scale is None:
<<<<<<< HEAD
            # if scale is not difinded, calculate scale as closest multiple number.
=======
            # スケールの定義
>>>>>>> 23dd86731ea147188fd490755d5fb2872295a946
            h_scale = (xh//chainer.config.user_multiple)*chainer.config.user_multiple/xh
            w_scale = (xw//chainer.config.user_multiple)*chainer.config.user_multiple/xw
            scale = h_scale, w_scale
        elif isinstance(scale, numbers.Number):
            scale = scale, scale
        elif isinstance(scale, tuple) and len(scale)>2:
            raise InvalidArgumentError

<<<<<<< HEAD
        new_sz = (int(xh*scale[0]), int(xw*scale[1]))  # specification of opencv, argments is recepted (w, h)
=======
        new_sz = (int(xh*scale[0]), int(xw*scale[1]))
>>>>>>> 23dd86731ea147188fd490755d5fb2872295a946
        image = transform.resize(image, new_sz, mode='constant')

        xh, xw = image.shape[:2]
        m0, m1 = xh % chainer.config.user_multiple, xw % chainer.config.user_multiple
        d0, d1 = np.random.randint(m0+1), np.random.randint(m1+1)

        image = image[d0:(image.shape[0] - m0 + d0), d1:(image.shape[1] - m1 + d1)]

        if len(image.shape)==2:
            return image.reshape((image.shape[0], image.shape[1], 1))
        else:
            return image, (m0, m1), (d0, d1)

    def flip(self, image, gt):
        do_flip_xy = np.random.randint(0, 2)
        do_flip_x = np.random.randint(0, 2)
        do_flip_y = np.random.randint(0, 2)

<<<<<<< HEAD
        if do_flip_xy: # Transpose X and Y axis
            image = image[::-1, ::-1, :]
            gt = gt[::-1, ::-1]
        elif do_flip_x: # Flip along Y-axis
            image = image[::-1, :, :]
            gt = gt[::-1, :]
        elif do_flip_y: # Flip along X-axis
=======
        if do_flip_xy: # X,Y軸の反転
            image = image[::-1, ::-1, :]
            gt = gt[::-1, ::-1]
        elif do_flip_x: # X軸の反転
            image = image[::-1, :, :]
            gt = gt[::-1, :]
        elif do_flip_y: # Y軸の反転
>>>>>>> 23dd86731ea147188fd490755d5fb2872295a946
            image = image[:, ::-1, :]
            gt = gt[:, ::-1]
        return image, gt,

    def scaling(self, image):
        do_scale = np.random.randint(0, 2)
        if do_scale:
            scale = self.args.aug_params.params.scale[ \
                np.random.randint(0,len(self.args.aug_params.params.scale))]
            return self.resize_image(image, scale)
        else:
            return image, None, None

    def random_brightness(self, image, max_delta=63, seed=None):
        brightness_flag = np.random.randint(0, 2)
        if brightness_flag:
            delta = np.random.uniform(-max_delta, max_delta)
            return image + delta
        else:
            return image

    def random_contrast(self, image, lower=0.2, upper=1.8, seed=None):
        contrast_flag = np.random.randint(0, 2)
        if contrast_flag:
            factor = np.random.uniform(-lower, upper)
            im_mean = image.mean(axis=2)
            return ((image.transpose(2, 0, 1) - im_mean)*factor + im_mean).transpose(1,2,0).astype(np.uint8)
        else:
            return image
<<<<<<< HEAD



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
=======
>>>>>>> 23dd86731ea147188fd490755d5fb2872295a946
