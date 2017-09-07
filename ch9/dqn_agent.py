import chainer.optimizers
import chainer.functions
import chainer.serializer
import gym
import math
import network
import numpy
import os
import random
import re
import sys

SEED_SYSTEM = 71
SEED_NUMPY = 71

random.seed(SEED_SYSTEM)
numpy.random.seed(SEED_NUMPY)


class Qfunc(object):
    def __init__(self, action_dim, state_dim, optimizer,
                 model_network=network.MLP3DQNet):
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        self.model = model_network(state_dim, action_dim)
        self.optimizer = optimizer
        self.optimizer.setup(self.model)

    def q_value(self, state_vector):
        tmp = self.model(state_vector).data
        return tmp

    def update(self, state_vector, action_vector, expected_returns):
        self.model.zerograds()
        q, loss = self.model(state_vector,
                             action_vector,
                             expected_returns.reshape(\
                             (len(expected_returns), 1)))
        loss.backward()
        self.optimizer.update()
        return loss.data

    def action(self, state_vector):
        q = self.q_value(state_vector)
        return numpy.argmax(q, axis=1)

    def update_interpol(self, target, mix_rate):
        self.model.weight_update((1-mix_rate), mix_rate, target)

    def save_model(self, outfname):
        chainer.serializers.save_npz(outfname, self.model)

    def load_model(self, infname):
        chainer.serializers.load_npz(infname, self.model)

        
class ReplayMemory(object):
    """
    Replay buffer class which provide to store (s, a, s', r, done flag) and to pick samples randomly.

    Args:
    state_dimension (int) : State space dimension
    action_dimension (int) : Number of possible actions
    max_size (int) : Size of replay buffer
    """
    def __init__(self, state_dim, action_dim, max_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.data = numpy.zeros((max_size, state_dim*2+action_dim+2), dtype=numpy.float32)
        self.max_size = max_size
        self.current = 0
        self.filled_size = 0

    def add(self, state_vector, action_vector, next_state_vector, reward, done):
        """
        Add a sample which consists of (s,a,r,s') and done flag.
        """
        self.data[self.current, 0:self.state_dim] = state_vector
        self.data[self.current, self.state_dim:self.state_dim+self.action_dim] = action_vector
        self.data[self.current, self.state_dim+self.action_dim:-2] = next_state_vector
        self.data[self.current, -2] = reward
        self.data[self.current, -1] = done

        self.current = (self.current+1)%self.max_size
        self.filled_size = min(self.filled_size+1, self.max_size)

    def select_samples(self, size):
        """
        Pick `size` samples randomly.
        
        Returns:
        tuple: state, action, next state, reward, done flag.
        If there are no sufficient amount of data, return a tuple of None.
        """
        if self.filled_size < size:
            return (None, None, None, None, None)

        sample_view = self.data[numpy.array(random.sample(range(self.filled_size), size))]
        return (sample_view[:, :self.state_dim],
                sample_view[:, self.state_dim:self.state_dim+self.action_dim],
                sample_view[:, self.state_dim+self.action_dim:-2],
                sample_view[:, -2],
                sample_view[:, -1])

class DQNGymAgent(object):
    def __init__(self, action_dim, state_dim,
                 memory_size=10000, batch_size=32, discount=0.99, mix_rate=0.001, 
                 optimizer=chainer.optimizers.Adam,
                 optimizer_args=[{}, {}],
                 model_network=network.MLP3DQNet):                 

        self.action_dim = action_dim
        self.state_dim = state_dim

        self.current_state = None
        self.prv_state = None
        self.prv_action = None
        
        self.discount = discount
        self.mix_rate = mix_rate
        self.batch_size = batch_size

        self.q_net = Qfunc(self.action_dim, self.state_dim, optimizer(**optimizer_args[0]), model_network))
        self.q_net_target = Qfunc(self.action_dim, self.state_dim, optimizer(**optimizer_args[0]), model_network)
        self.q_net_target.update_interpol(self.q_net.model, 1.0)

        self.memory = ReplayMemory(self.state_dim, self.action_dim, memory_size)
        
    def next_action(self, state, reward, done, info, e_greedy):
        self.prv_state = self.current_state
        self.current_state = state.astype(numpy.float32)
        self.add_sample(self.prv_state,
                        self.prv_action,
                        self.current_state,
                        reward, done)
        self.learn()
        self.prv_action = self.e_greedy(self.current_state, e_greedy)
        return self.prv_action
    
    def add_sample(self, state_vector, action, next_state_vector, reward, done):
        if state_vector is None or action is None or next_state_vector is None:
            return False
        else:
            action_vector = numpy.zeros((self.action_dim), numpy.float32)
            action_vector[action] = 1
            self.memory.add(state_vector,
                            action_vector,
                            next_state_vector,
                            reward, done)
            return True

    def e_greedy(self, state_vector, epsilon=0.001):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim-1)
        else:
            return self.q_net.action(state_vector.reshape((1,self.state_dim)))[0]

    def learn(self):
        (states,
         actions,
         next_states,
         reward, done) = self.memory.select_samples(self.batch_size)

        if states is None:
            return False
        
        expected_returns = (reward+
                            (1-done)*self.discount*
                            numpy.max(self.q_net_target.q_value(next_states), axis=1))

        loss = self.q_net.update(states, actions, expected_returns)
        self.q_net_target.update_interpol(self.q_net.model, self.mix_rate)

    def reset(self):
        """
        Reset previous episode info.
        """
        self.current_state = None

    def save_model(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.q_net.save_model(dir+'/Q_network.model')
        self.q_net_target.save_model(dir+'/Q_target_network.model')

    def load_model(self, dir):
        self.q_net.load_model(dir+'/Q_network.model')
        self.q_net_target.load_model(dir+'/Q_target_network.model')

        
if __name__=='__main__':
    environment = 'CartPole-v0' 
    n_episode = 200  # number of episodes
    max_iter = 200 # number of iterations

    discount = 0.99 # discount rate "gamma"
    epsilon = 0.05 # epilon for epsilon-greedy
    e_decay = 0.995 # decay rate for epsilon
    
    render = True

    env = gym.make(environment)
    agent = DQNGymAgent(env.action_space.n, env.observation_space.shape[0],
                        discount=discount, model_network=network.MLP3DQNet)

    for episode in range(n_episode):
        total_reward = 0
        reward = done = info = None
        observation = env.reset()
        for t in range(max_iter):
            if render:
                env.render()
            if reward is None:
                action = env.action_space.sample()
            else:
                action = agent.next_action(observation, reward, done, info, e_greedy=epsilon)

            observation, reward, done, info = env.step(action)

            total_reward += reward
            if done:
                agent.next_action(observation, reward, done, info, e_greedy=epsilon)
                agent.reset()
                break

        epsilon *= e_decay 
        print("{} {}".format(episode, total_reward))
