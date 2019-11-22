import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, BatchNormalization
from copy import copy
import matplotlib.pyplot as plt


class DQN:
    def __init__(self, number_episodes, training, double_gpu=False, freeze_rate=300, v_set=15):
        """Instantiate a RL Controller"""
        self.v_set = v_set
        self.a_set = []
        self.action_space = 3  # 3 discrete accelerations as actions
        self.feature_number = 2
        self.actions = np.linspace(-2., 2., self.action_space)  # possible actions (accelerations) of the NN controller
        self.index = 0  # placeholder for index of chosen action value
        self.action_value = np.zeros([1, 5])  # placeholder for the action values for all actions
        self.experience_batch_size = 100000  # number of samples in the batch
        self.experience_batch = np.zeros((1, 6), dtype=np.float64)  # Experience Replay batch,
        self.minibatch_size = 50  # number of samples in a minibatch used for 1 gradient descent step
        self.freeze_rate = freeze_rate
        self.update_frequency = 3  # The number of obtained transition tuples (time steps) before an update of Q is performed
        self.optimizer = tf.keras.optimizers.RMSprop(lr=0.0001)
        self.step_counter = 0
        self.minibatch = np.zeros((self.minibatch_size, 6), dtype=np.float64)
        self.double_gpu = double_gpu
        self.epsilon = np.zeros((number_episodes, 1))
        if training:  # no random actions when not training (--> epsilon = 0)
            self.epsilon = -np.arctan(
                (10 / number_episodes) * np.arange(-number_episodes / 2, number_episodes / 2)) * 1 / np.pi + 1 / 2
            # self.epsilon = self.epsilon / 2  # option to start from epsilon = ~0.5
            self.epsilon = np.ones([number_episodes, 1]) * 0.2
        self.state = np.zeros([1, self.feature_number], dtype=np.float64)
        self.new_state = np.zeros([1, self.feature_number], dtype=np.float64)
        self.endstate = False
        self.observed_weights = np.zeros([number_episodes, 6])

        # policy network
        with tf.device('/cpu:0'):
            self.model = tf.keras.Sequential()
            self.model.add(Dense(units=20, activation='relu',
                                 input_dim=self.feature_number))  # 2 input dimensions: distance, delta_v
            # self.model.add(BatchNormalization(axis=1))
            self.model.add(Dense(units=20, activation='relu', input_dim=20))
            # self.model.add(BatchNormalization(axis=1))
            self.model.add(Dense(units=self.action_space, activation='linear',
                                 input_dim=20))  # output dimensions: action values of n actions
        if self.double_gpu:
            self.parallel_model = tf.keras.utils.multi_gpu_model(self.model, gpus=2)
            self.parallel_model.compile(loss=self.clipped_mse, optimizer=self.optimizer)
        else:
            self.model.compile(loss=self.clipped_mse,
                               optimizer=self.optimizer)  # loss function: mean squared error (used for
            #  updates)

        # target network used to calculate the TD targets - updated every (freeze_rate) steps
        with tf.device('/cpu:0'):
            self.target_model = tf.keras.Sequential()
            self.target_model.add(
                Dense(units=20, activation='relu', input_dim=self.feature_number))  # 2 input dimensions: distance, delta_v
            # self.target_model.add(BatchNormalization(axis=1))
            self.target_model.add(Dense(units=20, activation='relu', input_dim=20))
            # self.target_model.add(BatchNormalization(axis=1))
            self.target_model.add(Dense(units=self.action_space, activation='linear',
                                        input_dim=20))  # output dimensions: action values of n actions
        if self.double_gpu:
            self.target_parallel_model = tf.keras.utils.multi_gpu_model(self.model, gpus=2)
            self.target_parallel_model.compile(loss=self.clipped_mse, optimizer=self.optimizer)
        else:
            self.target_model.compile(loss=self.clipped_mse,
                                      optimizer=self.optimizer)  # loss function: mean squared error (used for
            #  updates)

    def reset_variables(self):
        self.index = np.zeros((100000, 1), dtype=int)
        self.endstate = False

    def choose_action(self,Q_values, episode):
        """Returns the index of the action with highest action value (epsilon-greedy policy)"""
        if self.epsilon[episode] < np.random.random():
            a_index = np.argmax(Q_values)
        else:
            a_index = np.random.random(0, self.action_space)
        return a_index

    def predict_Q_values(self, state):
        """Calculates a forward pass of the NN to estimate the Q-values for the given state.
        Output is a vector of the Q-values for all possible actions"""
        Q_values = self.model.predict(state)
        return Q_values

    def update_Q(self, epochs, discount_factor, reward, state, state_new, a_index, endstate):
        """Calculate the Q-target (named "target_vector" here) and perform a NN update. The Q-target is just the Q values of the current state where
        the component corresponding to the chosen action in the current state is replaced by the
        TD target R+gamma*max(Q(s',a'). The Q target is the "desired output" of the NN. The error function (TD error) is the
        difference between the prediction of Q values of the current state and the Q target"""
        state = np.concatenate(state)
        state_new = np.concatenate(state_new)
        # Q_old = np.concatenate(Q_old)
        target_vector = self.target_model.predict(state)  # calculate the Q values for all actions of the current state
                                                          # --> the component corresponding to the chosen action is
                                                          # later replaced by the target
        for kk in range(len(self.minibatch)):
            if endstate[kk]:
                target = reward[kk]
            else:
                target = reward[kk] + discount_factor * np.max(self.target_model.predict(np.reshape(state_new[kk, :], (-1, self.feature_number))))  # scalar Q-target value, input to model (state_new)
                                                                                                                           # is reshaped to a (x,number_inputs) array
            # only the Q value for the chosen action should be changed, but keras expects the whole Q-value vector as output vector
            # --> target vector consists of the Q values of the current state, with the component of a_index changed to the actual target
            #target_vector[kk, :] = Q_old[kk, :]
            target_vector[kk, a_index[kk]] = target
        state = state.astype(np.float32)
        target_vector = target_vector.astype(np.float32)
        if self.double_gpu:
            self.parallel_model.fit(state, target_vector, epochs=epochs, verbose=0, batch_size=self.minibatch_size)
        else:
            self.model.fit(state, target_vector, epochs=epochs, verbose=0, batch_size=self.minibatch_size)

    def model_load(self, filename):
        """Load the pretrained model"""
        self.model = tf.keras.models.load_model(filename)

    def plot_Q_function(self):
        """ Calculate a matrix for the Q-values by calculating a forward pass for specific distance/v_diff input pairs"""
        input = np.zeros([1, self.feature_number])
        if self.feature_number == 1:
            Q_map = np.zeros((50, self.action_space))
            for v_ego in range(0, 50):
                input[0, 0] = v_ego
                input = input.astype(float)
                Q_map[v_ego, :] = self.model.predict(input)
        elif self.feature_number == 2:
            Q_map = np.zeros((500, 20, self.action_space))
            for distance in range(500):
                for delta_v in range(-10, 10):
                    input[0, 0] = distance
                    input[0, 1] = delta_v
                    Q_map[distance, delta_v, :] = self.model.predict(input)
        return Q_map

    def plot_best_policy(self, v_ego_min=0, v_ego_max=50, stepsize=0.1):
        input = np.zeros([1, self.feature_number])
        number_data_points = int((v_ego_max - v_ego_min) / stepsize)
        v_ego = np.arange(v_ego_min, v_ego_max, stepsize)
        if self.feature_number == 1:
            best_action_index = -np.ones([number_data_points, 1])
            for ii in range(number_data_points):
                input[0, 0] = v_ego[ii]
                input = input.astype(float)
                best_action_index[ii] = np.argmax(self.model.predict(input))
            best_action_index = best_action_index.astype(int)
            best_action = self.actions[best_action_index]
            plt.figure()
            plt.plot(v_ego, best_action)
            plt.xlabel('v_ego in m/s')
            plt.ylabel('a_set in m/s^2')
            plt.show(block=True)

    def reward_mean100(self, cumreward):
        """Calculate the respective mean of the last 100 rewards
        !!!to replace with the reward_mean_100_running function for simplicity!!!"""
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
        weights_all = self.model.get_weights()
        self.observed_weights[episode, 0] = weights_all[4][3, 0]  # 4th weight of 1st output neuron
        self.observed_weights[episode, 1] = weights_all[4][4, 1]  # 5th weight of 2nd output neuron
        self.observed_weights[episode, 2] = weights_all[4][5, 2]  # 6th weight of 3rd output neuron
        self.observed_weights[episode, 3] = weights_all[5][0]  # bias of 1st output neuron
        self.observed_weights[episode, 4] = weights_all[5][1]  # bias of 2nd output neuron
        self.observed_weights[episode, 5] = weights_all[5][2]  # bias of 3rd output neuron


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
        if self.double_gpu:
            weights = self.parallel_model.get_weights()
            self.target_parallel_model.set_weights(weights)
        else:
            weights = self.model.get_weights()
            self.target_model.set_weights(weights)
        print('Updated target network.')

    def clipped_mse(self, y_true, y_pred):
        """Mean squared error function with clipped difference of predicted and true value to (-1, 1)
        to help with robustness of the learning algorithm"""
        return tf.keras.backend.mean(tf.keras.backend.square(tf.keras.backend.clip(y_pred - y_true, -1., 1.)), axis=-1)

    def save_models(self, filename_model=None, filename_critic=None):
        """Save the models (structure, weights, biases) to .h5 files """
        self.model.save(filename_model)







