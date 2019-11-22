import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, BatchNormalization
from copy import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time
import time


class DDPG:
    """DDPG Controller. It uses an actor NN as policy pi(s|theta)(and a target actor NN for updates) to calculate a set acceleration
    and a critic NN Q(s,a|zeta) (plus a target critic NN) for estimating the Q function and to train the actor.
    Actor Input: states (e.g. distance, speed, ...)
    Actor Output: continuous action (acceleration)
    Critic Input: states, action
    Critic Output: Q-value for given action

    The critic is trained with the TD error [R+gamma*Q(s',pi_target(s')) - Q(s,a)]

    The actor is trained with the policy gradient [dQ/da * dpi/dtheta]
    """
    def __init__(self, number_episodes, training, feature_number, double_gpu=False, freeze_rate=1000, v_set=15, d_set=50, controller='DDPG'):
        """Instantiate the DDPG controller with all the NNs"""
        if controller == 'DDPG_v':
            self.action_range = np.array([-5., 25.]) # range for the values of the action (velocity values)
        else:
            self.action_range = np.array([-3, 3]) # range for the values of the action (acceleration values)
        self.v_set = v_set
        self.d_set = d_set
        self.a_set = []
        self.action_space = 1  # acceleration as single (continuous) action
        self.feature_number = feature_number
        self.experience_batch_size = 100000  # number of samples in the batch
        self.experience_batch = np.zeros((1, 6), dtype=np.float32)  # Experience Replay batch,
        self.minibatch_size = 64  # number of samples in a minibatch used for 1 gradient descent step
        self.freeze_rate = freeze_rate
        self.warmup_time = 2000  # wait for x time steps before NNs start getting updated
        self.update_frequency = 2  # The number of obtained transition tuples (time steps) before an update of Q is performed
        self.learning_rate_actor = 5.e-08  # 1.e-07
        self.learning_rate_critic = 5.e-07  # 1.e-06
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate_critic, clipnorm=1.)  # optimizer critic
        self.critic_loss = np.zeros((number_episodes*10000, 1))
        if controller == 'DDPG_v':
            self.actor_activation_output = self.scaled_sigmoid  # output of actor NN with range of (0, 30) (velocity)
        else:
            self.actor_activation_output = self.scaled_tanh  # output of actor NN with range of (-3, 3) (accelerations)
        self.relu_init = tf.keras.initializers.he_normal()
        self.tanh_init = tf.keras.initializers.lecun_normal()
        self.linear_init = tf.keras.initializers.glorot_normal()
        self.step_counter = 0
        self.minibatch = np.zeros((self.minibatch_size, 6), dtype=np.float32)
        self.double_gpu = double_gpu
        self.state = np.zeros([1, self.feature_number], dtype=np.float32)
        self.new_state = np.zeros([1, self.feature_number], dtype=np.float32)
        self.endstate = False
        self.index = np.zeros((1000000, 1), dtype=int)
        self.observed_weights = np.zeros([number_episodes, 6])
        self.OU_theta = 0.2  # mean reversion rate of the Ornstein Uhlenbeck process  --  0.5
        self.OU_mu = 8  # mean reversion level of the Ornstein Uhlenbeck process  --  0.1
        self.OU_sigma = 10  # diffusion coefficient of the Ornstein Uhlenbeck process  --  0.3
        self.OU_repeats = 2  # number of timesteps the delta_noise from OU is used for
        self.noise_counter = 0  # counting variable for OU repeat
        self.delta_noise = 0  # noise value added to calculated acceleration
        self.k_hybrid_a = 1/2  # factor for hybrid_a controller that scales the DDPG-set-acceleration
        self.weight_grad_sum = [0]


        # create the NN models
        self.sess = tf.Session()
        tf.keras.backend.set_session(self.sess)

        #self.optimizer = tf.keras.optimizers.RMSprop(lr=self.learning_rate_critic)

        # actor network
        with tf.device('/cpu:0'):
            self.actor_input = tf.keras.layers.Input(shape=(self.feature_number,))
            actor_hidden1 = tf.keras.layers.Dense(units=50, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer=self.relu_init)(self.actor_input)
            actor_hidden2 = tf.keras.layers.Dense(units=50, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer=self.relu_init)(actor_hidden1)
            actor_output = tf.keras.layers.Dense(units=self.action_space, activation=self.actor_activation_output, kernel_initializer=self.tanh_init)(actor_hidden2)
            actor_model = tf.keras.models.Model(inputs=self.actor_input, outputs=actor_output)
            if self.double_gpu:
                actor_parallel_model = tf.keras.utils.multi_gpu_model(actor_model, gpus=2)
                self.actor = actor_parallel_model
            else:
                self.actor = actor_model

        # actor target network
        self.target_actor = copy(self.actor)

        # critic network
        with tf.device('/cpu:0'):

            self.critic_input_state = tf.keras.layers.Input(shape=(self.feature_number,))
            self.critic_input_action = tf.keras.layers.Input(shape=(self.action_space,))
            critic_input = tf.keras.layers.concatenate([self.critic_input_state, self.critic_input_action])
            critic_hidden1 = tf.keras.layers.Dense(units=150, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer=self.relu_init)(critic_input)
            critic_hidden2 = tf.keras.layers.Dense(units=150, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer=self.relu_init)(critic_hidden1)
            critic_output = tf.keras.layers.Dense(units=1, activation='linear', kernel_initializer=self.linear_init)(critic_hidden2)
            critic_model = tf.keras.models.Model(inputs=[self.critic_input_state, self.critic_input_action], outputs=critic_output)
            """ critic topology with action feed-in in 2nd layer
            self.critic_input_state = tf.keras.layers.Input(shape=(self.feature_number,))
            self.critic_input_action = tf.keras.layers.Input(shape=(self.action_space,))
            critic_hidden1 = tf.keras.layers.Dense(units=150, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer=self.relu_init)(
                self.critic_input_state)
            critic_hidden1_with_action = tf.keras.layers.concatenate([critic_hidden1, self.critic_input_action])
            critic_hidden2 = tf.keras.layers.Dense(units=150, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer=self.relu_init)(
                critic_hidden1_with_action)
            critic_output = tf.keras.layers.Dense(units=1, activation='linear', kernel_initializer=self.linear_init)(critic_hidden2)
            critic_model = tf.keras.models.Model(inputs=[self.critic_input_state, self.critic_input_action], outputs=critic_output)
            """
            if self.double_gpu:
                critic_parallel_model = tf.keras.utils.multi_gpu_model(critic_model, gpus=2)
                critic_parallel_model.compile(loss='mse', optimizer=self.optimizer)  # loss=self.clipped_mse
                self.critic = critic_parallel_model
            else:
                critic_model.compile(loss='mse', optimizer=self.optimizer)  # loss=self.clipped_mse
                self.critic = critic_model

        # critic target network
        self.target_critic = copy(self.critic)

        # symbolic Gradient of Q w.r.t. a
        self.dQda = tf.gradients(self.critic.output, self.critic_input_action)

        # Tensorflow placeholder for dQda
        self.dQda_placeholder = tf.placeholder(tf.float32, [None, self.action_space])

        # symbolic Policy Gradient dpi/dtheta * dQ/da -- dQ/da fed into tf.gradients as multiplicative term grad_ys
        # minus sign in grad_ys to switch from a gradient descent to a gradient ascend formulation
        self.policy_gradient = tf.gradients(self.actor.output, self.actor.trainable_weights, grad_ys=-self.dQda_placeholder)

        # Optimizer method of the actor, inputs pairs of gradient (policy gradient) and weight
        grads_and_vars = zip(self.policy_gradient, self.actor.trainable_weights)
        self.optimize_actor = tf.train.AdamOptimizer(learning_rate=self.learning_rate_actor).apply_gradients(grads_and_vars)

        # Initialize tensorflow variables
        self.sess.run(tf.global_variables_initializer())

        #self.train_writer = tf.summary.FileWriter('logs', self.sess.graph)

        #self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time()))


    def update_actor(self, states, dQda):
        """ Update the actor network with the policy gradient. Update needs to be made 'manually' via tf command optimizer.apply_gradients
        (and not with model.fit) to be able to feed in the policy gradient"""
        dQda = np.concatenate(dQda)
        dQda = np.reshape(dQda, (-1, self.action_space))
        states = np.concatenate(states)
        states = np.reshape(states, (-1, self.feature_number))
        self.sess.run(self.optimize_actor,
                      feed_dict={self.actor_input: states, self.dQda_placeholder: dQda})

    def evaluate_dQda(self, states, actions):
        """Evaluate the gradient of Q w.r.t. a. States and actions are fed into the critic NN (which outputs Q)
        """
        states = np.concatenate(states)
        states = np.reshape(states, (-1, self.feature_number))
        actions = np.concatenate(actions)
        actions = np.reshape(actions, (-1, self.action_space))
        dQda = self.sess.run(self.dQda, feed_dict={self.critic_input_state: states, self.critic_input_action: actions})
        return dQda

    def reset_variables(self):
        self.index[:] = 0  # np.zeros((100000, 1), dtype=int)
        self.endstate = False

    def choose_action(self, state, network, features=(), noise=True):
        """Calculate a forward pass through the actor network to obtain a set acceleration"""
        state = np.concatenate(state)
        state = np.reshape(state, (-1, self.feature_number))
        if network == 'main':
            action = self.actor.predict(state)
        elif network == 'target':
            action = self.target_actor.predict(state)
        if noise:
            """Ornstein-Uhlenbeck Process"""
            noisy_action = self.add_noise(action, features)
            if noisy_action > np.amax(self.action_range):
                noisy_action = np.amax(self.action_range)
            elif noisy_action < np.amin(self.action_range):
                noisy_action = np.amin(self.action_range)
            return noisy_action, action
        else:
            return action

    def predict_Q(self, state, action):
        """Calculates a forward pass of the critic NN to estimate the Q-value for the given state
        TODO: Check Input format to NN"""
        state = np.reshape(state, (-1, self.feature_number))
        action = np.reshape(action, (-1, self.action_space))
        Q_value = self.critic.predict([state, action])
        return Q_value

    def update_critic(self, epochs, discount_factor, reward, state, state_new, action, endstate):
        """Calculate the Q-target (named "target_vector" here) and perform a NN update. The Q-target is the
        TD target R+(Q(s',pi(s')) for non-terminal states and R for terminal states. The Q target is the "desired output" of the NN. The error function
        (TD error) is the difference between the prediction of Q values of the current state and the Q target"""
        state = np.concatenate(state)
        state_new = np.concatenate(state_new)
        target = np.zeros([self.minibatch_size, 1])

        for kk in range(len(self.minibatch)):
            if endstate[kk]:
                target[kk] = reward[kk]
            else:
                target[kk] = reward[kk] + discount_factor * self.target_critic.predict((np.reshape(state_new[kk, :], (-1, self.feature_number)),
                    self.target_actor.predict(np.reshape(state_new[kk, :], (-1, self.feature_number)))))  # scalar Q-target value, input to model (state_new) is reshaped to a (x,number_inputs) array

                #target[kk] = reward[kk] + discount_factor * self.target_critic.predict([np.reshape(state_new[kk, :], (-1, self.feature_number))], [self.target_actor.predict(np.reshape(state_new[kk, :], (-1, self.feature_number)))])  # scalar Q-target value, input to model (state_new) is reshaped to a (x,number_inputs) array
        state = state.astype(np.float32)
        target = target.astype(np.float32)
        self.critic_loss[self.step_counter] = self.critic.train_on_batch((state, action), target)  # with callback: self.critic.fit((state, action), target, epochs=epochs, verbose=0, batch_size=self.minibatch_size, callbacks=[self.tensorboard])


    def model_load(self, filename_actor, filename_critic):
        """Load the pretrained model"""
        self.actor = tf.keras.models.load_model(filename_actor)
        self.critic = tf.keras.models.load_model(filename_critic)

    def plot_Q_function(self):
        """ Calculate a matrix for the Q-values by calculating a forward pass for specific feature input pairs"""
        input_state = np.zeros([1, self.feature_number])
        input_action = np.zeros([1, self.action_space])
        actions = np.linspace(-3., 3., 50)
        v_ego = np.linspace(0., 30., 50)
        if self.feature_number == 1:
            Q_map = np.zeros((len(v_ego), len(actions)))
            for v in range(len(v_ego)):
                for a in range(len(actions)):
                    input_state[0, 0] = self.v_set - v_ego[v]
                    input_state = input_state.astype(float)
                    input_action[0, 0] = actions[a]
                    Q_map[v, a] = self.critic.predict([input_state, input_action])
        elif self.feature_number == 2:
            """TODO: Adjust to DDPG critic layout"""
            Q_map = np.zeros((500, 20, self.action_space))
            for distance in range(500):
                for delta_v in range(-10, 10):
                    input[0, 0] = distance
                    input[0, 1] = delta_v
                    Q_map[distance, delta_v, :] = self.critic.predict(input)
        elif self.feature_number == 3:
            """TODO: Implementation"""
        return Q_map

    def plot_best_policy(self, v_ego_min=0, v_ego_max=50, stepsize=0.1, v_set=20, d_set=100):
        """TODO: Extend function for higher feature numbers"""
        input = np.zeros([1, self.feature_number])
        v_ego = np.arange(v_ego_min, v_ego_max, stepsize)
        distance = np.arange(0., 250., stepsize*20)
        v_diff = np.arange(-30., 30., stepsize*10)
        if self.feature_number == 1:
            best_action = np.ones([len(v_ego), 1])
            for ii in range(len(v_ego)):
                input[0, 0] = v_set - v_ego[ii]
                input = input.astype(float)
                best_action[ii] = self.actor.predict(input)
            plt.figure()
            plt.plot(v_ego, best_action)
            plt.xlabel('v_ego in m/s')
            plt.ylabel('a_set in m/s^2')
            plt.show(block=True)
        elif self.feature_number == 2:
            """feature space right now: e_distance, v_diff"""
            best_action = np.ones([len(distance), len(v_diff)])
            for ii in range(len(distance)):
                for jj in range(len(v_diff)):
                    input[0, 0] = d_set - distance[ii]
                    input[0, 1] = v_diff[jj]
                    best_action[ii, jj] = self.actor.predict(input)
            fig = plt.figure()
            X, Y = np.meshgrid(distance, v_diff)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, best_action.T)
            plt.show(block=True)
        elif self.feature_number == 3:
            """ Test implementation, only showing action as a function of distance and v_ego
            TODO: general implementation, how?"""
            best_action = np.ones([len(distance), len(v_ego)])
            for ii in range(len(distance)):
                for jj in range(len(v_ego)):
                    input[0, 0] = distance[ii]
                    input[0, 1] = v_ego[jj]
                    input[0, 2] = 10
                    best_action[ii, jj] = self.actor.predict(input)
            fig = plt.figure()
            X, Y = np.meshgrid(distance, v_ego)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, best_action.T)
            plt.show(block=True)

    def reward_mean100(self, cumreward):
        """Calculate the respective mean of the last 100 rewards
        TODO: replace with the reward_mean_100_running function for simplicity"""
        cumreward_mean100 = np.zeros(np.size(cumreward))
        for ii in range(len(cumreward)):
            if ii <= 100:
                cumreward_mean100[ii] = np.mean(cumreward[1:ii])
            else:
                cumreward_mean100[ii] = np.mean(cumreward[ii - 100:ii])
        return cumreward_mean100

    def reward_mean_100_running(self, cumreward, episode):
        """Calculate the respective mean of the last 100 rewards"""
        if episode == 0:
            cumreward_mean100 = cumreward[0]
        elif episode <= 100 and episode > 0:
            cumreward_mean100 = np.mean(cumreward[:episode+1])
        else:
            cumreward_mean100 = np.mean(cumreward[episode - 100:episode])
        return cumreward_mean100

    def weight_observer(self, episode):
        """Read the values of some weights and biases of the NN during training for plotting"""
        weights_all = self.actor.get_weights()
        self.observed_weights[episode, 0] = weights_all[4][3, 0]  # 4th weight of output neuron actor
        self.observed_weights[episode, 1] = weights_all[4][4, 0]  # 5th weight of output neuron actor
        self.observed_weights[episode, 2] = weights_all[4][5, 0]  # 6th weight of output neuron actor
        self.observed_weights[episode, 3] = weights_all[5][0]  # bias of output neuron actor



    def add_sample(self, state, action, state_new, reward, endstate, episode):
        """Add a sample (state,action,new_state,reward) to the experience replay batch"""
        new_sample = np.array([state, action, state_new, reward, endstate])
        if self.step_counter == 0 and episode == 0:
            self.experience_batch = new_sample
            self.experience_batch = np.vstack([self.experience_batch, new_sample])  # first sample twice in the batch to be able to index over the rows
        elif len(self.experience_batch) < self.experience_batch_size:
            self.experience_batch = np.vstack([self.experience_batch, new_sample])  # add new sample to batch when it is not full
        else:
            self.experience_batch[self.step_counter % self.experience_batch_size, :] = new_sample  # override the components of the batch when it is full

    def create_minibatch(self):
        """Sample a random minibatch of samples. The size of the batch is given in the __init__ method"""
        if self.experience_batch.shape[0] <= self.minibatch_size:
            self.minibatch = self.experience_batch

        else:
            ind = np.random.randint(self.experience_batch.shape[0], size=self.minibatch_size)  # same sample can be in the minibatch multiple times --> problem for algorithm ?
            self.minibatch = self.experience_batch[ind]

    def predict_next_state(self, state, action, stepsize, tau, a_previous):
        """ Try to predict the next state of the MDP based on the current state and action chosen.
        Besides the current state and action, the prediction is based on the known system dynamics a_real = f(a_set)
        + a predictor to estimate the acceleration of the preceding vehicle"""
        distance = state[1]
        delta_v = state[2]
        a_real = (action + tau * a_previous / stepsize) / (1 + tau/stepsize)  # a_real to replace with more detailed longitudinal dynamics model
        distance_next = distance + stepsize * delta_v + (stepsize^2)/2 * a_real - (stepsize^2)/2 - (stepsize^2)/2 * a_prec  # a_prec to replace with NN
        delta_v_next = delta_v + stepsize*(a_real - a_prec)
        return distance_next, delta_v_next

    def update_target_network(self):
        weights_actor = self.actor.get_weights()
        self.target_actor.set_weights(weights_actor)
        weights_critic = self.critic.get_weights()
        self.target_critic.set_weights(weights_critic)
        print('Updated target networks.')

    def clipped_mse(self, y_true, y_pred):
        """Mean squared error function with clipped difference of predicted and true value to (-1, 1)
        to help with robustness of the learning algorithm"""
        return tf.keras.backend.mean(tf.keras.backend.square(tf.keras.backend.clip(y_pred - y_true, -1., 1.)), axis=-1)

    def save_models(self, filename_actor=None, filename_critic=None):
        self.actor.save_weights(filename_actor+'.h5')
        self.critic.save_weights(filename_critic+'.h5')

    def scaled_tanh(self, x):
        """Scaled tanh function to interval (-3,3)"""
        return tf.keras.backend.tanh(x) * 3

    def scaled_sigmoid(self, x):
        """Scaled sigmoid function to interval (0, 30)"""
        return (tf.keras.backend.sigmoid(x) * 30 - 5)

    def add_noise(self, action, features):
        if self.noise_counter == 0:
            self.set_OU_param(features)
            """Add noise according to an Ornstein-Uhlenbeck process"""
            self.delta_noise = self.OU_theta * (self.OU_mu - action) + self.OU_sigma * np.random.randn()
            self.noise_counter = self.OU_repeats
        else:
            self.noise_counter -= 1
        """Test weighted sum of Ornstein-Uhlenbeck noise + v_ego proportional noise"""
        noisy_action = action + self.delta_noise
        if noisy_action > np.amax(self.action_range):
            noisy_action = np.amax(self.action_range)
        elif noisy_action < np.amin(self.action_range):
            noisy_action = np.amin(self.action_range)
        return noisy_action

    def supervised_actor_training(self, state_data, action_data):
        state_data = np.reshape(state_data, (-1, self.feature_number))
        action_data = np.reshape(action_data, (-1, self.action_space))
        supervised_actor = copy(self.actor)
        supervised_optimizer = tf.keras.optimizers.Adam(lr=0.001)
        supervised_actor.compile(optimizer=supervised_optimizer, loss='mse')
        supervised_actor.fit(state_data, action_data, epochs=100, batch_size=50)
        supervised_actor.save_weights('saved_models/supervised_init_weights_distance_control.h5')

    def set_OU_param(self, features):
        self.OU_mu = 0.6*features.v_allowed
        self.OU_sigma = 0.75 * features.v_allowed






