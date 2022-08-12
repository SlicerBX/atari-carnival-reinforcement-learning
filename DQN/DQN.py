import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from time import time
import datetime
import cv2

state_size = (96, 80, 1)


# AVG time for 176x128 is 9 minutes (I turned up my PC's performance). 11.6 million parameters. Score was 500-ish
# AVG time for 128x96 is  10.75 minutes. 6.4 million parameters AVG SCORE is 580
# AVG time for 96x80 is  7.5 minutes. 4 million parameters AVG SCORE is 752.
# Likely less time variance with the smaller frames. Let's go small.

class DQN_Agent:
    # Initializes attributes and constructs CNN model and target_model

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # Replay memory

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
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.model.summary()

    #
    # Constructs CNN
    #
    def _build_model(self):
        model = Sequential()

        # Conv Layers
        # The first number is the output depth. The tuples are kernel size.
        model.add(Conv2D(32, (8, 8), strides=4, padding='same', input_shape=self.state_size, activation='relu'))
        model.add(Conv2D(64, (4, 4), strides=2, padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(Flatten())

        # FC Layers
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.add(Flatten())
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    # Stores experience in replay memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Chooses action based on epsilon-greedy policy
    def act(self, state):
        # Random exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)

        return np.argmax(act_values[0])  # Returns action using policy

    # Trains the model using randomly selected experiences in the replay memory
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)  # Batch size is the return size

        for state, action, reward, next_state, done in minibatch:

            if not done:
                target = (reward + self.gamma * np.amax(self.target_model.predict(next_state)))  # TD
            else:
                target = reward

            # Construct the target vector as follows:
            # Use the current model to output the Q-value predictions.
            target_f = self.model.predict(state)  # This constructs our Q value

            # Rewrite the chosen action value with the computed target
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)  # Train model so Q-value match the target function

        #  Simple decay rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Decay rate

    # Sets the target model parameters to the current model parameters
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


'''Preprocessing: To save on computations'''


# Helpful preprocessing taken from github.com/ageron/tiny-dqn
def process_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = img / 255  # Normalize
    img = cv2.resize(img, state_size[:2]) # Using CV ensure the use of proper rbg weights
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
    checkpoint_filepath = './DQN_weights/checkpoint'
    env = gym.make('Carnival-v0')

    action_size = env.action_space.n
    agent = DQN_Agent(state_size, action_size)
    episodes = 5
    list_of_episode_rewards = []
    batch_size = 8
    skip_start = 90  # MsPacman-v0 waits for 90 actions before the episode begins
    total_time = 0  # Counter for total number of steps taken
    all_rewards = 0  # Used to compute avg reward over time
    blend = 4  # Number of images to blend
    done = False

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/Carnivaldqn/' + current_time
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
        for t in range(1200):  # We aim to maximize the score in the first 5000 steps

            # Return the avg of the last 4 frames.
            state = blend_images(images, blend)

            # Transition
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
        agent.save('./DQN_ckpt/Carnival96x80')

    env.close()