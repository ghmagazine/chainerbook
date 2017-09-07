import chainer
import chainer.links
import chainer.functions


class MLP3DQNet(chainer.Chain):
    def __init__(self, insize, outsize):
        super(MLP3DQNet, self).__init__()
        with self.init_scope():
            self.layer1 = chainer.links.Linear(insize, 30)
            self.layer2 = chainer.links.Linear(30, 30)
            self.layer3 = chainer.links.Linear(30, outsize)

    def __call__(self, state, action=None, t=None):
        h = chainer.functions.relu(self.layer1(state))
        h = chainer.functions.relu(self.layer2(h))
        h = self.layer3(h)

        if t is not None:
            q = chainer.functions.reshape(chainer.functions.batch_matmul(h, action, transa=True), t.shape)
            return h, chainer.functions.mean_squared_error(q, t)                                                          
        else:
            return h

    def weight_update(self, w1, w2, operand):
        op_params = dict(operand.namedparams())
        for name, param in self.namedparams():
            param.data = w1*param.data+w2*op_params[name].data


class MLP2DQNet(chainer.Chain):
    def __init__(self, insize, outsize):
        super(MLP2DQNet, self).__init__()
        with self.init_scope():
            self.layer1 = chainer.links.Linear(insize, 500)
            self.layer2 = chainer.links.Linear(500, outsize)

    def __call__(self, state, action=None, t=None):
        h = chainer.functions.relu(self.layer1(state))
        h = self.layer2(h)

        if t is not None:
            q = chainer.functions.reshape(chainer.functions.batch_matmul(h, action, transa=True), t.shape)
            return h, chainer.functions.mean_squared_error(q, t)                                                          
        else:
            return h

    def weight_update(self, w1, w2, operand):
        op_params = dict(operand.namedparams())
        for name, param in self.namedparams():
            param.data = w1*param.data+w2*op_params[name].data


class MLP1DQNet(chainer.Chain):
    def __init__(self, insize, outsize):
        super(MLP2DQNet, self).__init__()
        with self.init_scope():
            self.layer1 = chainer.links.Linear(insize, outsize)

    def __call__(self, state, action=None, t=None):
        h = chainer.functions.relu(self.layer1(state))

        if t is not None:
            q = chainer.functions.reshape(chainer.functions.batch_matmul(h, action, transa=True), t.shape)
            return h, chainer.functions.mean_squared_error(q, t)                                                          
        else:
            return h

    def weight_update(self, w1, w2, operand):
        op_params = dict(operand.namedparams())
        for name, param in self.namedparams():
            param.data = w1*param.data+w2*op_params[name].data
            
