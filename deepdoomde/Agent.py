#!/usr/bin/env python3
'''
Agent.py
Author: Rafael Zamora
Updated: 5/16/17

'''

# Import Packages
import deepdoomde
import json, sys, os
import vizdoom as vzd
import itertools as it
from contextlib import contextmanager
import numpy as np
from tqdm import tqdm
from random import sample
import keras.backend as K
K.set_image_data_format("channels_first")
from keras.models import Model
from keras.layers import *
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from vis.visualization import visualize_cam

class DoomAgent:
    """
    Class is used to interface agent with ViZDoom. This class is used to train,
    test, run agents in Vizdoom Scenarios.

    This agent trains skills using Double Deep Q-Learning and
    then combines skills through the use of h-DRL and Policy Distillation.

    """

    def __init__(self):
        '''
        '''
        self.data_path = os.path.expanduser('~') + '/.deepdoomde/'
        self.module_path = os.path.dirname(deepdoomde.__file__)

        # Initiate ViZDoom
        self.vizdoom = vzd.DoomGame()
        self.vizdoom.set_doom_game_path(self.module_path + "/deepdoom.wad")
        self.vizdoom.load_config(self.module_path + "/agent_config.cfg")
        self.vizdoom.set_window_visible(False)
        self.vizdoom.init()
        self.all_actions = self.vizdoom.get_available_buttons()
        self.vizdoom.close()

        # Parameters and Variables
        self.enviro_config = None
        self.agent_model = None
        self.input = None
        self.input_size = 1

    def agent_info(self):
        '''
        Method displays agent configuration info.

        '''
        self.agent_model.agent_info()
        print("Neural Network Architecture:")
        self.agent_model.online_network.summary()

    def load_agent(self, config_file, load_weights=True, weight_file=None):
        '''
        Method loads agent model defined within the configuration file.

        '''
        self.agent_model = AgentModel(self.vizdoom, self.all_actions)
        self.agent_model.load(config_file)
        self.input_size = self.agent_model.get_frames()
        if load_weights:
            if weight_file: self.agent_model.load_weights(weight_file)
            else:
                if os.path.isfile(config_file[:-4]+'.h5'):
                    self.agent_model.load_weights(config_file[:-4]+'.h5')

    def set_enviro(self, config_file):
        '''
        Method sets environment to the defined configuration file.

        '''
        self.enviro_config = config_file

    def load_enviro(self, render=False, doom_like=False, total_tics=None):
        '''
        Method loads environment from set .cfg file.

        '''
        self.vizdoom.set_doom_game_path(self.module_path + "/deepdoom.wad")
        self.vizdoom.load_config(self.module_path + "/agent_config.cfg")
        self.vizdoom.load_config(self.enviro_config)
        self.vizdoom.set_window_visible(False)
        if total_tics: self.vizdoom.set_episode_timeout(total_tics)
        if render:
            self.vizdoom.set_screen_resolution(vzd.ScreenResolution.RES_800X600)
            self.vizdoom.set_window_visible(True)
            if doom_like:
                self.vizdoom.set_render_hud(True)
                self.vizdoom.set_render_minimal_hud(False)
                self.vizdoom.set_render_crosshair(False)
                self.vizdoom.set_render_weapon(True)
                self.vizdoom.set_render_particles(True)

        with suppress_stdout():
            self.vizdoom.init()

    def get_state(self):
        '''
        Method updates state data available to the agent.

        '''
        if not self.input: self.input = [np.zeros((2, 120, 160)) for i in range(self.input_size)]
        state = self.vizdoom.get_state()
        screen_buffer = np.expand_dims(state.screen_buffer, 0)
        depth_buffer = np.expand_dims(state.depth_buffer, 0)
        if depth_buffer.any(): depth_buffer = 255 - depth_buffer
        else: depth_buffer = np.zeros((1, 120, 160))
        s = np.concatenate([screen_buffer, depth_buffer])
        self.input.append(s)
        self.input.pop(0)
        S = np.expand_dims(self.input, 0)
        return S

    def train(self, epochs, steps, memory_size, batch_size, gamma, target_update, nb_tests, alpha_decay=False, weight_file=None, print_graph=None):
        '''
        Method runs training session on agent model using Double Deep Q-Learning.
        Epsilon decays begins after the first 1/10 of epochs and reduces to 0.1

        '''
        # Initiate Models
        self.agent_model.target_network = Model(inputs=self.agent_model.x0, outputs=self.agent_model.y0)
        self.agent_model.target_network.set_weights(self.agent_model.online_network.get_weights())
        self.agent_model.target_network.compile(optimizer=self.agent_model.optimizer, loss=self.agent_model.loss_fun)

        # Set Training Parameters
        epsilon = 1.0
        alpha = 1.0
        memory = []
        training_data = []
        best_score = None

        # Training Loop
        title = self.agent_model.config.split('/')[-1] + ' on ' + self.enviro_config.split('/')[-1]
        print("Training:", title)
        for epoch in range(epochs):
            self.load_enviro()
            pbar = tqdm(total=steps)
            step, loss, total_reward, a_prime = 0, 0, 0, 0
            self.vizdoom.new_episode()
            S = self.get_state()

            # Preform learning step
            while step < steps:

                # Agent's Action and Reward Response
                argmax_q, r = self.agent_model.make_action(self.vizdoom, S, epsilon)

                # Store transition in memory
                a = argmax_q
                game_over = self.vizdoom.is_episode_finished()
                if game_over: S_prime = np.zeros((1,self.agent_model.nb_frames,2,120,160))
                else: S_prime = self.get_state()
                memory.append(np.concatenate([self.agent_model.process_input(S).flatten(),
                    np.array(a).flatten(), np.array(r).flatten(), self.agent_model.process_input(S_prime).flatten(),
                    np.array(a_prime).flatten(), 1 * np.array(game_over).flatten()]))
                if memory_size > 0 and len(memory) > memory_size: memory.pop(0)
                S = S_prime
                a_prime = a

                # Generate training batch
                nb_actions = self.agent_model.nb_actions
                input_dim = np.prod((self.agent_model.nb_frames,120,160))
                if len(memory) < batch_size: batch_size_ = len(memory)
                else: batch_size_ = batch_size
                samples = np.array(sample(memory, batch_size_))

                ## Restructure Data
                s_ = samples[:, 0 : input_dim]
                a_ = samples[:, input_dim]
                r_ = samples[:, input_dim + 1]
                s_prime_ = samples[:, input_dim + 2 : 2 * input_dim + 2]
                a_prime_ = samples[:, 2 * input_dim + 2]
                game_over_ = samples[:, 2 * input_dim + 3]
                r_ = r_.repeat(nb_actions).reshape((batch_size_, nb_actions))
                game_over_ = game_over_.repeat(nb_actions).reshape((batch_size_, nb_actions))
                s_ = s_.reshape((batch_size_, ) + (self.agent_model.nb_frames,120,160))
                s_prime_ = s_prime_.reshape((batch_size_, ) + (self.agent_model.nb_frames,120,160))

                ## Predict Q-Values
                X = np.concatenate([s_, s_prime_], axis=0)
                Y = self.agent_model.online_network.predict(X)
                best = np.argmax(Y[batch_size_:], axis = 1)
                YY = self.agent_model.target_network.predict(s_prime_)
                Qsa = YY.flatten()[np.arange(batch_size_)*nb_actions + best].repeat(nb_actions).reshape((batch_size_, nb_actions))
                delta = np.zeros((batch_size_, nb_actions))
                a_ = np.cast['int'](a_)
                delta[np.arange(batch_size_), a_] = 1

                ## Get target Q-Values
                targets = ((1 - delta) * Y[:batch_size_]) + ((alpha * ((delta * (r_ + (gamma * (1 - game_over) * Qsa))) - (delta * Y[:batch_size_]))) + (delta * Y[:batch_size_]))

                # Train Agent Model's Online Network
                loss += float(self.agent_model.online_network.train_on_batch(s_, targets))

                # Update Target Network Weights
                if self.agent_model.target_network and step % target_update == 0:
                    self.agent_model.target_network.set_weights(self.agent_model.online_network.get_weights())

                if game_over:
                    self.vizdoom.new_episode()
                    S = self.get_state()
                step += 1
                pbar.update(1)

            self.vizdoom.close()

            # Run Tests
            print("Testing:")
            pbar.close()
            pbar = tqdm(total=nb_tests)
            total_rewards = []
            for i in range(nb_tests):
                total_reward = self.test()
                total_rewards.append(total_reward)
                pbar.update(1)
            total_rewards = np.array(total_rewards)
            training_data.append([loss, np.mean(total_rewards), np.max(total_rewards), np.min(total_rewards)])
            if print_graph:
                t = np.array(training_data)
                plt.plot(t[:,3], color='#e6e6e6'); plt.plot(t[:,2], color='#e6e6e6')
                plt.fill_between(list(range(len(t[:,3]))), t[:,3],t[:,2],interpolate=True,color='#e6e6e6')
                plt.plot(t[:,1], color='blue'); plt.title('Training: '+title, fontsize=12)
                plt.ylabel('Average Reward Per Epoch'); plt.xlabel('Training Epochs')
                plt.savefig(print_graph); plt.figure()

            # Save Best Weights
            total_reward_avg = training_data[-1][1]
            if best_score is None or (best_score is not None and total_reward_avg > best_score):
                if weight_file: self.agent_model.save_weights(weight_file)
                else: self.agent_model.save_weights(self.agent_model.config[:-4] + ".h5")
                best_score = total_reward_avg

            # Print Epoch Summary
            print("Epoch {:03d}/{:03d} | Loss {:.4f} | Epsilon {:.3f} | Average Total Reward {}".format(
            epoch + 1, epochs, loss, epsilon, total_reward_avg))

            # Decay Epsilon
            if epsilon > 0.1 and epochs > 1 and epoch >= int(epochs/10):
                epsilon -= 0.9 / int((epochs/10) * 8)
                if alpha_decay: alpha -= 0.9 / int((epochs/10) * 8)

        print("Training Finished.\nBest Average Reward:", best_score)

    def distill(self, epochs, steps, memory_size, batch_size, gamma, target_update, nb_tests, overwrite=True, save_data=True):
        '''
        '''

        # Set Training Parameters
        epsilon = 1.0
        memory = []
        training_data = []
        best_score = None

        # Training Loop
        title = self.agent_model.config[:-4] + ' on ' + self.enviro_config
        print("Training:", title)
        for epoch in range(epochs):
            self.load_enviro()
            pbar = tqdm(total=steps)
            step, loss, total_reward, a_prime = 0, 0, 0, 0
            self.vizdoom.new_episode()
            S = self.get_state()

            # Preform learning step
            while step < steps:

                # Agent's Action and Reward Response
                argmax_q, r = self.agent_model.make_action(self.vizdoom, S, epsilon)

                action, argmax_q, q = self.get_action(S)

                inputs = []
                targets = []
                inputs.append(S[0])

                # Advance Action over frame_skips + 1
                if not self.vizdoom.is_episode_finished():
                    self.vizdoom.set_action(action)
                    self.vizdoom.advance_action(self.input_skips+1, True)

                # Advance Action over skill_frame_skip
                if argmax_q < len(self.agent_model.actions):
                    for i in range(self.skill_frame_skip):
                        action, temp = self.get_action(S, argmax_q)
                        if not self.vizdoom.is_episode_finished():
                            self.vizdoom.set_action(action)
                            self.vizdoom.advance_action(self.input_skips+1, True)
                        else: break

				# Train policy model online network
                loss += float(self.agent_model.online_network.train_on_batch(inputs, targets))

                if game_over:
                    self.vizdoom.new_episode()
                    S = self.get_state_data(game)
                step += 1
                pbar.update(1)

            # Decay Epsilon
            if self.epsilon > 0.1 and epoch >= 10: self.epsilon -= 0.0125

            # Run Tests
            print("Testing:")
            pbar.close()
            pbar = tqdm(total=20)
            total_rewards = []
            for i in range(20):
                total_reward, last_reward = self.test()
                total_rewards.append(total_reward)
                pbar.update(1)
            total_rewards = np.array(total_rewards)
            training_data.append([loss, np.mean(total_rewards), np.max(total_rewards), np.min(total_rewards)])
            np.savetxt("../data/results/distlled_" + self.routine_file[:-4] + ".csv", np.array(training_data))

			# Save best weights
            total_reward_avg = training_data[-1][1]
            if best_score is None or (best_score is not None and total_reward_avg > best_score):
                self.agent_model.save_weights("distilled_" + self.routine_file[:-4] + ".h5")
                best_score = total_reward_avg

			# Print Epoch Summary
            print("Epoch {:03d}/{:03d} | Loss {:.4f} | Epsilon {:.3f} | Average Total Reward {}".format(
            epoch + 1, self.nb_epoch, loss, self.epsilon, total_reward_avg))

        print("Training Finished.\nBest Average Reward:", best_score)

    def test(self, save_replay=None, verbose=False, visualization=False, timeout=None):
        '''
        Method runs a test of VizDoom instance.

        '''
        self.input = None

        # Initiate Vizdoom instance
        self.load_enviro(total_tics=timeout)
        if save_replay: self.vizdoom.new_episode(save_replay)
        else: self.vizdoom.new_episode()

        if verbose:
            title = self.agent_model.config.split('/')[-1] + ' on ' + self.enviro_config.split('/')[-1]
            print("Running Test:", title)
            pbar = tqdm(total=self.vizdoom.get_episode_timeout())

        if visualization:
            qs = []
            q_ = []
            s_ = []
            hs_ = []
            r_ = []

        # Run Test
        while not self.vizdoom.is_episode_finished():
            S = self.get_state()
            if verbose: i = int(self.vizdoom.get_state().tic)
            if visualization:
                s = self.agent_model.process_input(S)
                q = self.agent_model.online_network.predict(s)
                blockPrint()
                heatmap = visualize_cam(self.agent_model.online_network, 6, [np.argmax(q[0])], s[0].transpose(1,2,0), alpha=0.0)
                heatmap = np.dot(heatmap[...,:3], [0.299, 0.587, 0.114])
                enablePrint()
                qs.append(softmax(q[0],1.0))
                q_.append(q[0])
                s_.append(s[0][-1])
                hs_.append(heatmap)
                r_.append(self.vizdoom.get_total_reward())
            self.agent_model.make_action(self.vizdoom, S)
            if verbose:
                if not self.vizdoom.is_episode_finished():
                    d = int(self.vizdoom.get_state().tic) - i
                pbar.update(d)
        if visualization:
            plt.ion()
            fig = plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k', tight_layout=True)
            qs_ = np.array(qs)
            q_ = np.array(q_)
            for i in range(1, len(qs)):
                plt.gcf().clear()
                plt.subplot2grid((3, 4), (0, 0))
                plt.axis('off')
                plt.title("Processed Input", fontsize=12, fontweight='bold')
                plt.imshow(s_[i],cmap="gray", interpolation="nearest")
                plt.subplot2grid((3, 4), (0, 1))
                plt.axis('off')
                plt.title("Attention Map", fontsize=12, fontweight='bold')
                plt.imshow(hs_[i], cmap='gist_heat', interpolation="gaussian")
                plt.colorbar()
                plt.subplot2grid((3, 4), (2, 0), colspan=3)
                plt.axis('on')
                plt.xlabel('State')
                plt.ylabel('Q-Values')
                for j in range(len(qs_[i])):
                    y = q_[:i, j]
                    plt.plot([k for k in range(len(y))], y, label=str(self.agent_model.actions[j]))
                plt.subplot2grid((3, 4), (1, 0), colspan=3)
                plt.axis('on')
                plt.title("Predicted Q-Values", fontsize=12, fontweight='bold')
                plt.xlabel('State')
                plt.ylabel('Q-Values (Softmax)')
                for j in range(len(qs_[i])):
                    y = qs_[:i, j]
                    plt.plot([k for k in range(len(y))], y, label=str(self.agent_model.actions[j]))
                plt.legend(loc='upper left', bbox_to_anchor=(1.05, 2.4), title='Action Combos')
                plt.figtext(0.575, 0.95, "Agent: " + self.agent_model.config[:-4], fontsize=12, fontweight='bold')
                plt.figtext(0.575, 0.90, "Enviro: " + self.enviro_config, fontsize=12, fontweight='bold')
                plt.figtext(0.575, 0.85, "Input Skips: " + str(self.agent_model.input_skips), fontsize=12, fontweight='bold')
                plt.figtext(0.575, 0.80, "Depth Contrast: " + str(self.agent_model.depth_contrast), fontsize=12, fontweight='bold')
                plt.figtext(0.575, 0.75, "Nb Frames: " + str(self.agent_model.nb_frames), fontsize=12, fontweight='bold')
                plt.figtext(0.575, 0.70, "Total Score: " + str(r_[i]), fontsize=12, fontweight='bold')
                fig.canvas.draw()

        # Reset Agent and Return Score
        score = self.vizdoom.get_total_reward()
        if verbose: pbar.close(); print("Total Score:", score)
        self.vizdoom.close()
        return score

    def replay(self, filename, doom_like=False):
        '''
        Method runs a replay of the simulations at 800 x 600 resolution.

        '''
        print("Running Replay:", filename.split('/')[-1])

        # Initiate Replay
        self.load_enviro(render=True, doom_like=doom_like)
        self.vizdoom.replay_episode(filename)

        # Run Replay
        while not self.vizdoom.is_episode_finished():
            self.vizdoom.advance_action()
        self.vizdoom.close()

class AgentModel:
    """
    """

    def __init__(self, vizdoom, all_actions):
        '''
        '''
        # Model Parameters
        self.vizdoom = vizdoom
        self.all_actions = [str(i) for i in all_actions]
        self.av_actions = []
        self.config = ''
        self.actions = []
        self.sub_agents = []
        self.nb_actions = 0
        self.nb_frames = 1
        self.depth_contrast = 0.0
        self.input_skips = 0

        # Network Parameters
        self.loss_fun = 'mse'
        self.optimizer = RMSprop(lr=0.0001)

        # Input/Output Layers
        self.x0 = None
        self.y0 = None

        # Networks
        self.online_network = None
        self.target_network = None

    def load(self, config_file):
        '''
        Method loads agent configuration using defined file.

        '''
        # Load Model Config Json
        with open(config_file) as cfile: config = json.load(cfile)
        self.config = config_file
        self.av_actions = config['actions']
        act_codes = [list(a) for a in it.product([0, 1], repeat=len(self.all_actions))]
        self.actions = []
        for act in self.av_actions:
            if act.endswith('.agt'):
                sub_agent = AgentModel(self.vizdoom, self.all_actions)
                sub_agent.load(act)
                self.sub_agents.append(sub_agent)

        for ac in act_codes:
            flag = True
            for i in range(len(ac)):
                if ac[i] == 1:
                    if self.all_actions[i] not in self.av_actions:
                        flag = False
                        break
            if flag: self.actions.append(ac)

        self.nb_actions = len(self.actions) + len(self.sub_agents)
        self.depth_contrast = config['depth_contrast']
        self.nb_frames = config['nb_frames']
        self.input_skips = config['input_skips']
        self.compile()

    def compile(self, distilled=False):
        '''
        Method compiles Deep Q-Network.

        '''
        # Input Layers
        self.x0 = Input(shape=(self.nb_frames, 120, 160))

        # Convolutional Layer
        m = Conv2D(32, (8, 8), strides = (4,4), activation='relu')(self.x0)
        m = Conv2D(64, (5, 5), strides = (4,4), activation='relu')(m)
        m = Flatten()(m)

        # Fully Connected Layer
        m = Dense(4032, activation='relu')(m)
        m = Dropout(0.5)(m)

        # Output Layer
        if distilled: self.y0 = Dense(self.nb_actions, activation='softmax')(m)
        else: self.y0 = Dense(self.nb_actions + len(self.sub_agents))(m)

        self.online_network = Model(inputs=self.x0, outputs=self.y0)
        self.online_network.compile(optimizer=self.optimizer, loss=self.loss_fun)

    def process_input(self, S):
        '''
        Method processed input by combining greyscaled buffer with depth buffer
        using defined contrast value.

        '''
        processed_S = []
        s = S.astype('float')/255
        for i in range(len(s[0])):
            # Only use the number of frames needed
            if i >= len(s[0]) - self.nb_frames:
                processed_s = ((1 - self.depth_contrast) * s[0][i][0]) + (self.depth_contrast * s[0][i][1])
                processed_s = (processed_s - np.amin(processed_s))/ (np.amax(processed_s) - np.amin(processed_s) + 0.000001)
                processed_s = np.round(processed_s, 6)
                processed_S.append(processed_s)
        return np.expand_dims(processed_S, 0)

    def make_action(self, vizdoom, S, epsilon=0.1):
        '''
        Method select best action from policy models.

        '''
        # Process Input
        s = self.process_input(S)
        q = self.online_network.predict(s)
        r = 0

        # Exploration Policy
        if np.random.random() < epsilon:
            argmax_q = np.random.randint(len(q[0]))
            #argmax_q = np.random.choice(len(q[0]), 1, p=softmax(q[0], 1))[0]
        else: argmax_q = np.random.choice(len(q[0]), 1, p=softmax(q[0], 1))[0]#argmax_q = int(np.argmax(q[0]))

        # Select Action
        if argmax_q >= len(self.sub_agents):
            # Argmax Q is a native action
            a = argmax_q - len(self.sub_agents)
            vizdoom.set_action(self.actions[a])
            vizdoom.advance_action(self.input_skips+1, True)
            r += vizdoom.get_last_reward()
        else:
            # Argmax Q is a sub agent
            sub_agent = self.sub_agent[argmax_q]
            temp, sub_r = sub_agent.make_action(vizdoom, S, epsilon)
            r += sub_r

        return argmax_q, r

    def get_frames(self):
        '''
        Method returns how many frames of input the agent requires to run policy.

        '''
        nb_frames = self.nb_frames
        for a in self.sub_agents:
            x = a.get_frames()
            if x > nb_frames: nb_frames = x
        return nb_frames

    def agent_info(self, str_=''):
        '''
        Method displays agent configuration info.

        '''
        print(str_+"Agent Config:", self.config.split('/')[-1])
        print(str_+"Depth Contrast:", self.depth_contrast)
        print(str_+"Nb of Frames:", self.nb_frames)
        print(str_+"Input Skips:", self.input_skips)
        print(str_+"Available Actions:"); [print(str_+'\t'+i) for i in self.av_actions if not i.endswith('.agt')]
        print(str_+"Sub-Agents:")
        for i in range(len(self.sub_agents)):
            print(str_+"Sub-Agent",i)
            self.sub_agents[i].agent_info(str_=str_+'\t')

    def load_weights(self, filename):
        '''
        Method loads DQN model weights from file.

        '''
        self.online_network.load_weights(filename)
        self.online_network.compile(optimizer=self.optimizer, loss=self.loss_fun)

    def save_weights(self, filename):
        '''
        Method saves DQN model weights to file.s

        '''
        self.online_network.save_weights(filename, overwrite=True)

def blockPrint(): sys.stdout = open(os.devnull, 'w')

def enablePrint(): sys.stdout = sys.__stdout__


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
sys.stdout = old_stdout

def softmax(x, t): e_x = np.exp(x - np.max(x))/t; return e_x / e_x.sum(axis=0)
