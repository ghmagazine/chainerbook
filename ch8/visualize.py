import chainer
import chainer.links
import chainer.functions
import chainer.function_hooks
import chainer.serializers
from sklearn.datasets import fetch_mldata
import numpy
import sys
import PIL.Image
import ram_network


if __name__ == '__main__':
    numpy.random.seed(71)

    print("preparing dataset...", file=sys.stderr)
    mnist = fetch_mldata("MNIST original")
    mnist.data = mnist.data.astype(numpy.float32)
    mnist.data = mnist.data.reshape(mnist.data.shape[0], 1, 28, 28)
    mnist.target = mnist.target.astype(numpy.int32)
    train_data, test_data = numpy.split(mnist.data, [60000], axis=0)
    train_targets, test_targets = numpy.split(mnist.target, [60000])

    train_data /= 255
    test_data /= 255

    model = ram_network.RAM()

    chainer.serializers.load_npz('saved_model', model)

    with chainer.no_backprop_mode():
        label, positions = model.predict(test_data)

    crop_size=8
    p = numpy.array(positions)
    for i, l in enumerate(test_targets):
        with open('out/{}.txt'.format(i), 'w') as f:
            f.write('Correct label: {}\n'.format(l))
            f.write('Predicted label sequence:')
            for j in range(6):
                f.write('{} '.format(label[j][i]))
            f.write('\n')                
        img = numpy.concatenate([test_data[i].copy() for _ in range(6)], axis=2)
        for j in range(6):
            h = test_data[i].shape[1]
            w = test_data[i].shape[2]
            y_top = int((h-crop_size)*(positions[j][i,0]+1)//2)
            x_left = int((w-crop_size)*(positions[j][i,1]+1)//2)+w*j

            img[:, y_top, x_left: x_left+8] = 1.0
            img[:, y_top+7, x_left: x_left+8] = 1.0                
            img[:, y_top:y_top+8, x_left] = 1.0
            img[:, y_top:y_top+8, x_left+7] = 1.0

        img *= 255
        img = img.astype(numpy.uint8)
        img = img.transpose((1,2,0))
        img = img.reshape((img.shape[0], img.shape[1]))

        PIL.Image.fromarray(img).save('out/{}.jpg'.format(i,j))
