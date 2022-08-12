import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,  Flatten, Conv2D, MaxPooling2D
from time import time
import datetime
import math


import cv2


tf.keras.backend.set_floatx('float32')
tf.config.run_functions_eagerly(True)
# session = tf.Session()
# tf.compat.v1.disable_eager_execution()

state_size = (96, 80, 1)


# AVG time for 176x128 is 9 minutes (I turned up my PC's performance). 11.6 million parameters. Score was 500-ish
# AVG time for 128x96 is  10.75 minutes. 6.4 million parameters AVG SCORE is 580
# AVG time for 96x80 is  7.5 minutes. 4 million parameters AVG SCORE is 752.
# Likely less time variance with the smaller frames. Let's go small.
'''
To aid in the development and understanding of the necessary Tensorflow framework, we used reference code. 
https://github.com/marload/DistRL-TensorFlow2/blob/master/IQN/IQN.py
Which was intended for CartPole and not Atari Games. 
'''


class IQN(tf.keras.Model):
    def __init__(self, action_size, batch_size):
        super(IQN, self).__init__()
        self.action_size = action_size

        # N denotes the respective # of iid samples tau. Used to estimate the loss
        self.embedding_size = 64
        self.tau_sample_size = 8  # Number of quantile samples
        self.kernel_initializer = tf.keras.initializers.HeNormal(seed=None)
        self.batch_size = batch_size

        self.huber_loss = tf.keras.losses.Huber(
            reduction=tf.keras.losses.Reduction.NONE)
        self.opt = tf.compat.v1.train.RMSPropOptimizer(
            learning_rate=0.00025,
            decay=0.95,
            momentum=0.0,
            epsilon=0.00001,
            centered=True)

        self.pi = tf.constant(math.pi)
        # phi = ReLu(sum{cos(pi*i*tau)}w_ij + b)

        self.conv1 = Conv2D(
            32, kernel_size=[8, 8], strides=4, padding='same', activation='relu',
            kernel_initializer=self.kernel_initializer, name='Conv')
        self.conv2 = Conv2D(
            64, kernel_size=[4, 4], strides=2, padding='same', activation='relu',
            kernel_initializer=self.kernel_initializer, name='Conv')
        self.conv3 = Conv2D(
            64, kernel_size=[3, 3], strides=1, padding='same', activation='relu',
            kernel_initializer=self.kernel_initializer, name='Conv')

        self.flatten = Flatten()
        self.dense1 = Dense( 512, activation='relu', kernel_initializer=self.kernel_initializer, name='fully_connected')
        self.dense2 = Dense(self.action_size, kernel_initializer=self.kernel_initializer, name='fully_connected')
        self.max_pooling = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")

    def call(self, state):
        batch_size = state.get_shape().as_list()[0]
        # Z_tau(x,a) = f( \phi(x) hadamardProduct phi(tau)  )
        x = tf.cast(state, tf.float32)  # For some reason not all my tensors are in full precision.

        # Phi(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.max_pooling(x)
        x = self.flatten(x)
        state_vector_length = x.get_shape().as_list()[-1]

        # Create the quantile layer in the first call. This is because
        # number of output units depends on the input shape. Therefore, we can only
        # create the layer during the first forward call
        self.dense_quantile = Dense(state_vector_length, activation='relu', kernel_initializer=self.kernel_initializer)

        state_net_tiled = tf.tile(x, [self.tau_sample_size, 1])  # Tile expands our input x
        # self.tau_sample_size in the x direction and 1 in the y direction

        quantiles_shape = [self.tau_sample_size * batch_size, 1]
        quantiles = tf.random.uniform(
            quantiles_shape, minval=0, maxval=1, dtype=tf.float32)
        quantile_net = tf.tile(quantiles, [1, self.embedding_size])

        # sum{cos(pi*i*tau)}w_ij + b
        quantile_net = tf.cast(tf.range(1, self.embedding_size + 1, 1), tf.float32) * self.pi * quantile_net
        quantile_net = tf.cos(quantile_net)

        # F( \psi, \phi )
        quantile_net = self.dense_quantile(quantile_net)
        x = tf.multiply(state_net_tiled, quantile_net)

        x = self.dense1(x)
        quantile_values = self.dense2(x)
        return quantile_values, quantiles


class ActionValueModel:
    def __init__(self, state_size, action_size, batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.embedding_size = 32
        self.tau_sample_size = 8

        self.huber_loss = tf.keras.losses.Huber(
            reduction=tf.keras.losses.Reduction.NONE)

        self.opt = tf.compat.v1.train.RMSPropOptimizer(
            learning_rate=0.00025,
            decay=0.95,
            momentum=0.0,
            epsilon=0.00001,
            centered=True)
        self.model = self.create_model()

    def create_model(self):
        network = IQN(self.action_size, self.batch_size)
        network.build((1, state_size[0], state_size[1], state_size[2]))
        return network

    # Custom loss function.
    def quantile_huber_loss(self, target, pred, actions, tau):
        actions = tf.cast(actions, 'float32')
        pred = tf.reduce_sum(pred * tf.expand_dims(actions, -1), axis=1)
        pred_tile = tf.tile(tf.expand_dims(pred, axis=1), [1, self.model.tau_sample_size])
        target_tile = tf.tile(tf.expand_dims(
            target, axis=1), [1, self.model.tau_sample_size])
        target_tile = tf.cast(target_tile, 'float32')
        huber_loss = self.model.huber_loss(target_tile, pred_tile)
        tau = tf.reshape(np.array(tau), [self.model.tau_sample_size])
        inv_tau = 1.0 - tau
        tau = tf.tile(tf.expand_dims(tau, axis=1), [1, self.model.tau_sample_size])
        inv_tau = tf.tile(tf.expand_dims(inv_tau, axis=1), [1, self.model.tau_sample_size])
        error_loss = tf.math.subtract(target_tile, pred_tile)
        loss = tf.where(tf.less(error_loss, 0.0), inv_tau *
                        huber_loss, tau * huber_loss)
        loss = tf.reduce_mean(tf.reduce_sum(
            tf.reduce_mean(loss, axis=1), axis=0))
        return loss

    def train(self, states, target, actions):
        # Backwards pass
        with tf.GradientTape() as tape:
            theta, tau = self.model(states)
            loss = self.quantile_huber_loss(target, theta, actions, tau)
        grads = tape.gradient(loss, self.model.trainable_variables)
        # We keep on adding nodes when back propagating the gradients
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        tf.compat.v1.reset_default_graph()

    def predict(self, state):
        return self.model(state, training=False)


class IQNAgent:
    def __init__(self, state_size, action_size, batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=5000)  # Replay memory

        # Hyperparameters
        self.gamma = .99  # Discount rate
        self.epsilon = .99  # Exploration rate
        self.epsilon_min = 0.1  # Minimal exploration rate (epsilon-greedy)
        self.epsilon_decay = 0.999  # Decay rate for epsilon
        self.update_rate = 4  # Number of steps until updating the target network

        # CNN Hyperparameters
        self.learning_rate = 0.00025
        self.loss = tf.compat.v1.losses.huber_loss
        self.optimizer = tf.compat.v1.train.RMSPropOptimizer(
            learning_rate=0.00025,
            decay=0.95,
            momentum=0.0,
            epsilon=0.00001,
            centered=True)

        # Construct DQN models
        self.model = ActionValueModel(self.state_size, self.action_size, self.batch_size)
        self.target_model = ActionValueModel(self.state_size, self.action_size, self.batch_size)
        self.update_target_model()
        # tf.compat.v1.get_default_graph().finalize()
        # self.model.summary()

    # Stores experience in replay memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Chooses action based on epsilon-greedy policy
    def act(self, state):
        # Random exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)

        return tf.argmax(tf.reduce_mean(act_values[0], axis=0), axis=0)  # E(Z(x,\tau))

    # Trains the model using randomly selected experiences in the replay memory
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)  # Batch size is the return size
        for state, action, reward, next_state, done in minibatch:

            q, tau = self.target_model.predict(next_state)  # tau
            next_action = np.argmax(np.mean(q, axis=0), axis=0)
            theta = []
            for i in range(self.batch_size):
                if done:
                    theta.append(reward)  # Originally it was np.ones(self.model.tau_sample_size)*reward. Idk why
                    # If we reach a teminal state we get the reward for moving and there is nothing else to do.
                else:
                    theta.append(reward + self.gamma * q[i][next_action])
            # self.model.train(state, theta, action)
            # session.run(self.model.train(state, theta, action))

        # Probably need to fix this decay rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Decay rate

    def quantile_huber_loss(self, target, pred, actions, tau):
        actions = tf.cast(actions, 'float32')
        pred = tf.reduce_sum(pred * tf.expand_dims(actions, -1), axis=1)
        pred_tile = tf.tile(tf.expand_dims(pred, axis=1), [1, self.model.tau_sample_size])
        target_tile = tf.tile(tf.expand_dims(
            target, axis=1), [1, self.model.tau_sample_size])
        target_tile = tf.cast(target_tile, 'float32')
        huber_loss = self.model.huber_loss(target_tile, pred_tile)
        tau = tf.reshape(np.array(tau), [self.model.tau_sample_size])
        inv_tau = 1.0 - tau
        tau = tf.tile(tf.expand_dims(tau, axis=1), [1, self.model.tau_sample_size])
        inv_tau = tf.tile(tf.expand_dims(inv_tau, axis=1), [1, self.model.tau_sample_size])
        error_loss = tf.math.subtract(target_tile, pred_tile)
        loss = tf.where(tf.less(error_loss, 0.0), inv_tau *
                        huber_loss, tau * huber_loss)
        loss = tf.reduce_mean(tf.reduce_sum(
            tf.reduce_mean(loss, axis=1), axis=0))
        return loss

    # Sets the target model parameters to the current model parameters
    def update_target_model(self):
        self.target_model.model.set_weights(self.model.model.get_weights())

    # Loads a saved model
    def load(self, name):
        self.model.model.load_weights(name)

    # Saves parameters of a trained model
    def save(self, name):
        self.model.model.save_weights(name)


def process_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = img / 255  # Normalize
    img = cv2.resize(img, state_size[:2])
    return np.expand_dims(img.reshape(state_size), axis=0)


def blend_images(images, blend):
    avg_image = np.expand_dims(np.zeros(state_size, np.float64), axis=0)

    for image in images:
        avg_image += image

    if len(images) < blend:
        return avg_image / len(images)
    else:
        return avg_image / blend


if __name__ == '__main__':
    checkpoint_filepath = './IQN_weights/checkpoint'
    env = gym.make('Carnival-v0')

    episodes = 25
    list_of_episode_rewards = []
    batch_size = 8
    skip_start = 90  # MsPacman-v0 waits for 90 actions before the episode begins
    total_time = 0  # Counter for total number of steps taken
    all_rewards = 0  # Used to compute avg reward over time
    blend = 4  # Number of images to blend
    done = False

    action_size = env.action_space.n
    agent = IQNAgent(state_size, action_size, batch_size)

    tau = 1e-3

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/IQN/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    # An episode is essentially an Epoch.
    for e in range(episodes):
        start = time()
        episode_reward = 0
        game_score = 0
        state = process_frame(env.reset())
        images = deque(maxlen=blend)  # Array of images to be blended
        images.append(state)
        done = False
        for t in range(1200):  # Reducing steps determine how long an episode can take.

            # Return the avg of the last 4 frames
            state = blend_images(images, blend)

            # Transition Dynamics
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)

            # Return the avg of the last 4 frames
            next_state = process_frame(next_state)
            images.append(next_state)
            next_state = blend_images(images, blend)

            # Store sequence in replay memory
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            game_score += reward
            episode_reward += reward

            if done:
                all_rewards += game_score
                list_of_episode_rewards.append(episode_reward)
                avg_rewards = np.mean(list_of_episode_rewards[max(0, e - 10):(e + 1)])  # Moving average
                end_time = (time() - start)
                total_time += end_time
                print(
                    "episode: {}/{}, actions_taken: {} ,game score: {}, reward: {}, avg reward (5 episodes): {}, time: {:.2f}, total time: {:.2f}, epsilon: {:.4f}"
                        .format(e + 1, episodes, t, game_score, episode_reward, avg_rewards, end_time, total_time,
                                agent.epsilon))

                break

            if len(agent.memory) > 1000:  # This line will take up most computational time
                agent.replay(batch_size)

            # Every update_rate timesteps we update the target network parameters
            if t % agent.update_rate == 0:
                agent.update_target_model()

        if not done:
            all_rewards += game_score
            list_of_episode_rewards.append(episode_reward)
            avg_rewards = np.mean(list_of_episode_rewards[max(0, e - 5):(e + 1)])  # Moving average
            end_time = (time() - start)
            total_time += end_time
            print(
                "episode: {}/{}, actions_taken: {} ,game score: {}, reward: {}, avg reward (5 episodes): {}, time: {:.2f}, total time: {:.2f}, epsilon: {:.4f}"
                    .format(e + 1, episodes, t, game_score, episode_reward, avg_rewards, end_time, total_time,
                            agent.epsilon))

        with summary_writer.as_default():
            tf.summary.scalar('episode reward', episode_reward, step=e)
            tf.summary.scalar('running avg reward(5)', avg_rewards, step=e)
            tf.summary.scalar('number of steps', t, step=e)
            tf.summary.scalar('Time this episode', end_time, step=e)
        agent.save('./IQN_ckpt/Carnival96x80')

    env.close()
