import chainer
import chainer.links
import chainer.functions
import chainer.function_hooks
import chainer.serializers
from sklearn.datasets import fetch_mldata
import numpy
import sys
import PIL.Image


class RAM(chainer.Chain):
    def __init__(self, n_class=10, n_embeded=128, n_hidden=256, n_step=6, crop_size=8, variance=0.01):
        super(RAM, self).__init__()
        with self.init_scope():
            # position to vector
            self.pos_to_vec_layer = chainer.links.Linear(2, n_embeded)
            # cropped image to vector
            self.img_to_vec_layer = chainer.links.Linear(crop_size*crop_size, n_embeded)
            # fc-layer for pos vector
            self.fc_pos_vec_layer = chainer.links.Linear(n_embeded, n_hidden)
            # fc-layer for img vector
            self.fc_img_vec_layer = chainer.links.Linear(n_embeded, n_hidden)
            self.lstm = chainer.links.LSTM(n_hidden, n_hidden)
            self.make_position = chainer.links.Linear(n_hidden, 2)
            self.predict_label = chainer.links.Linear(n_hidden, n_class)
            self.estimate_baseline = chainer.links.Linear(n_hidden, 1)
        self.crop_size = crop_size
        self.n_step = n_step
        self.variance = variance
        
    def __call__(self, imgs, labels):
        self.lstm.reset_state()
        position = numpy.zeros((imgs.shape[0],2), dtype=numpy.float32)
        sum_lp = 0
        # move on n_step
        for i in range(self.n_step):
            position, lp, predicts, baseline = self.forward_one_step(imgs, position)
            sum_lp += lp
        return self.policy_baseline_loss(predicts, labels, baseline, sum_lp)\
            +self.label_loss(predicts, labels), chainer.functions.accuracy(predicts, labels)

    def predict(self, imgs):
        self.lstm.reset_state()
        position = numpy.zeros((imgs.shape[0],2), dtype=numpy.float32)
        sum_lp = 0
        out_p = []
        out_l = []
        # move on n_step
        out_p.append(position.copy())
        for i in range(self.n_step):
            position, _, predict, _ = self.forward_one_step(imgs, position)
            out_l.append(numpy.argmax(predict.data,axis=1))
            out_p.append(numpy.array(position.data).copy())
        return out_l, out_p

    def forward_one_step(self, imgs, position):
        glimpse = self.glimpse(imgs, position)
        hidden = self.lstm(glimpse)
        position, mean_position  = self.propose_position(hidden)
        lp = self.cal_logprob(mean_position, position, self.variance)
        predicts = self.predict_label(hidden)
        baseline = chainer.functions.reshape(chainer.functions.clip(\
                    self.estimate_baseline(hidden), 0., 1.),(-1,))
        return  position, lp, predicts, baseline

    def glimpse(self, imgs, position):
        cropped_imgs = self.crop(imgs, position, self.crop_size)
        h_imgs = self.img_to_vec_layer(cropped_imgs)
        h_imgs = chainer.functions.relu(h_imgs)
        h_imgs = self.fc_img_vec_layer(h_imgs)
        h_position = chainer.functions.relu(self.pos_to_vec_layer(position))
        h_position = self.fc_img_vec_layer(h_position)
        g = chainer.functions.relu(h_imgs+h_position)
        return g

    def propose_position(self, hidden):
        mean = chainer.functions.clip(self.make_position(hidden), -1., 1.)
        position = mean+numpy.random.normal(0., numpy.sqrt(self.variance), size=mean.shape).astype(numpy.float32)
        position = chainer.functions.clip(position, -1., 1.).data
        return position, mean

    def policy_baseline_loss(self, predicts, labels, baseline, logprob):
        batch_size = predicts.shape[0]
        reward = numpy.where(numpy.argmax(predicts.data,axis=1)==labels.data,
                             1,
                             0).astype(numpy.float32)
        baseline_loss = chainer.functions.mean_squared_error(reward, baseline)
        with chainer.no_backprop_mode():
            policy_loss = (1e-1)*chainer.functions.sum(-logprob*(reward-baseline))/batch_size
        return baseline_loss+policy_loss

    def label_loss(self, predicts, labels):
        return chainer.functions.softmax_cross_entropy(predicts, labels)

    @staticmethod
    def cal_logprob(location_mean, sampled_location, var):
        logprob = -0.5*(sampled_location - location_mean)**2/var
        logprob = chainer.functions.sum(logprob, axis=1)
        return logprob

    @staticmethod
    def crop(x, position, crop_size):
        h = x.shape[2]
        w = x.shape[3]
        y_top = ((h-crop_size)*(position[:,0]+1)//2).astype(int)
        x_left = ((w-crop_size)*(position[:,1]+1)//2).astype(int)

        y = numpy.zeros((x.shape[0], x.shape[1], crop_size, crop_size), dtype=numpy.float32)

        for k in range(x.shape[0]):
            y[k,:,:,:] = x[k, :, y_top[k]:y_top[k]+crop_size, x_left[k]:x_left[k]+crop_size]
        return y

    
if __name__=='__main__':
    print("preparing dataset...", file=sys.stderr)
    mnist = fetch_mldata("MNIST original")
    mnist.data = mnist.data.astype(numpy.float32)
    mnist.data = mnist.data.reshape(mnist.data.shape[0], 1, 28, 28)
    mnist.target = mnist.target.astype(numpy.int32)
    train_data, test_data = numpy.split(mnist.data, [60000], axis=0)
    train_targets, test_targets = numpy.split(mnist.target, [60000])

    train_data /= 255
    test_data /= 255

    model = RAM()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5))
    model.cleargrads()

    n_epoch = 10
    n_data = train_data.shape[0]
    batchsize = 128

    for epoch in range(n_epoch):
        perm = numpy.random.permutation(n_data)
        for i in range(0, n_data, batchsize):
            train_batch = train_data[perm[i:i+batchsize]]
            label_batch = train_targets[perm[i:i+batchsize]]
            loss, acc = model(train_batch, label_batch)
            loss.backward()
            loss.unchain_backward() # truncate chain for lstm
            optimizer.update()
            model.cleargrads()
            print('train loss:', loss.data, 'train acc:', acc.data)
        with chainer.no_backprop_mode():
            loss, acc = model(test_data, test_targets)
        print('test loss:', loss.data, 'test acc:', acc.data)

    chainer.serializers.save_npz('saved_model', model)
