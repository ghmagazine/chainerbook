import os
from easydict import EasyDict as edict
import chainer


<<<<<<< HEAD
# paths
data_root_path = './chapter6_0shimax/data'
=======
# パスの定義
data_root_path = './data'
>>>>>>> 23dd86731ea147188fd490755d5fb2872295a946
image_dir_path = os.path.join(data_root_path, 'CamSeq01')
train_image_pointer_file_name = 'train_image_pointer'
test_image_pointer_file_name = 'test_image_pointer'
output_path = os.path.join(data_root_path, 'result/fcn_dilated')
label_path = os.path.join(data_root_path, 'label_color')
chainer.global_config.user_multiple = 16
model_path = os.path.join(output_path, 'model/fcn_dilated_{}.model')

<<<<<<< HEAD
# training parameters
training_params = {
    # 'optimizer': 'RMSpropGraves',  # 'NesterovAG',
=======
# トレーニングパラメタ
training_params = {
>>>>>>> 23dd86731ea147188fd490755d5fb2872295a946
    'learning_rate': 1e-4,
    'batch_size': 20,
    'weight_decay': 0.0005,
    'epochs': 1,
    'snapshot_epochs': 5,
}

<<<<<<< HEAD
# hand making params
number_of_class = 32
debug_mode = False
in_ch = 3
gpu = -1


=======
# オーグメンテーションパラメタ
>>>>>>> 23dd86731ea147188fd490755d5fb2872295a946
augmentation_params = {'scale':[0.25, 0.5],}
aug_flags = {'do_scale':False,
             'do_flip':True,
             'change_britghtness':False,
             'change_contrast':False}

<<<<<<< HEAD

# a body of args
=======
# その他のパラメタ
number_of_class = 32
debug_mode = False
in_ch = 3
gpu = -1

# argsの定義
>>>>>>> 23dd86731ea147188fd490755d5fb2872295a946
train_args = \
    {
        'train': True,
        'debug_mode': debug_mode,
        'gpu': gpu,
        'n_class': number_of_class,
        'in_ch': in_ch,
        'image_dir_path': image_dir_path,
        'image_pointer_path': os.path.join(data_root_path, train_image_pointer_file_name),
        'output_path': output_path,
        'model_path': model_path,
        'aug_params': {'do_augment':True,
                       'params': dict(augmentation_params, **aug_flags),
                      },
        'training_params': training_params,
        'label_path': label_path,
    }

test_args = \
    {
        'train': False,
        'debug_mode': debug_mode,
        'gpu': gpu,
        'n_class': number_of_class,
        'in_ch': in_ch,
        'image_dir_path': image_dir_path,
        'image_pointer_path': os.path.join(data_root_path, test_image_pointer_file_name),
        'output_path': output_path,
        'model_path': model_path,
        'aug_params': {'do_augment':False},
        'label_path': label_path,
        'model_version': 'final',
    }


chainer.global_config.user_train_args = edict(train_args)
chainer.global_config.user_test_args = edict(test_args)
<<<<<<< HEAD

# def get_args(args_type='train'):
#     if args_type=='train':
#         return edict(train_args)
#     else:
#         return edict(test_args)
=======
>>>>>>> 23dd86731ea147188fd490755d5fb2872295a946
