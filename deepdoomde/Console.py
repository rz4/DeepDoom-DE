#!/usr/bin/env python3
'''
Console.py
Author: Rafael Zamora
Updated: 05/16/17

'''

# Import Packages
import os, sys, getopt
from cmd import Cmd
import datetime

class Console(Cmd):
    """
    Class runs DeepDoom-DE console used to run agent training sessions and tests.

    """

    def __init__(self, verbose=True):
        super().__init__()
        self.verbose = verbose
        self.intro = '\nWelcome to DeepDoom-DE! For more information use the \'help\' command.'
        self.prompt = '>>> '
        self.log = []
        if self.verbose:
            # Display Banner
            print("*****************************************************************")
            print("______               ______                        ______ _____ ")
            print("|  _  \              |  _  \                       |  _  \  ___|")
            print("| | | |___  ___ _ __ | | | |___   ___  _ __ ___    | | | | |_   ")
            print("| | | / _ \/ _ \ '_ \| | | / _ \ / _ \| '_ ` _ \   | | | |  __| ")
            print("| |/ /  __/  __/ |_) | |/ / (_) | (_) | | | | | |  | |/ /| |__  ")
            print("|___/ \___|\___| .__/|___/ \___/ \___/|_| |_| |_|  |___/ \____/ ")
            print("               | | ")
            print("               |_|  Version 0.1.0, Atlas Software 2017\n")
            print("Deep Reinforcement Learning Development Environment for Doom-Bots")
            print("\n\t\t\t   Powered By ViZDoom 1.1.1 and Keras 2.0")
            print("*****************************************************************")
            print("Starting DeepDoom-DE...")

        from deepdoomde.Agent import DoomAgent
        self.agent = DoomAgent()

    def do_loadagent(self, args):
        '''Command loads agent from specified .atg file.
        \nOptions:
        \n-u,--untrained\tWeights will not be loaded.
        \n-w,--weights\t<weight file to be loaded>'''

        untrained = False
        weights = None

        # Get Options
        opts, argss = getopt.getopt(args.split(), "uw:", ['untrained', 'weights='])
        for opt, arg in opts:
            if opt in ('-u', '--untrained'): untrained = True
            elif opt in ('-w', '--weights'): weights = arg
        argss = argss[0]

        # Find Agent File
        if os.path.isfile(argss): fname = argss
        elif os.path.isfile(self.agent.data_path + "agents/" + argss):
            fname = self.agent.data_path + "agents/" + argss
        else: print("Error: Cannot Find Agent Config: ", argss); return

        # Load Agent
        self.agent.load_agent(fname, load_weights=not untrained, weight_file=weights)
        self.log.append("loadagent " + args)

    def complete_loadagent(self, text, line, begidx, endidx):
        agent_files = [i for i in sorted(os.listdir(self.agent.data_path+"agents"))
                        if i.startswith(text) and i.endswith('.agt')]
        agent_files += [i for i in sorted(os.listdir(os.getcwd()))
                        if i.startswith(text) and i.endswith('.agt')]
        return agent_files

    def do_agentinfo(self, args):
        '''Command displays configuration information about the currently loaded agent.
        \nNote: Agent must be loaded first.'''

        if self.agent.agent_model:
            print("Displaying Loaded Agent Info...")
            self.agent.agent_info()
            self.log.append("agentinfo " + args)
        else: print("Error: No Agent Loaded.")

    def do_loadenviro(self, args):
        '''Command loads environment from specified .cfg file.'''

        # Find Enviro File
        if os.path.isfile(args): fname = args
        elif os.path.isfile(self.agent.data_path + "enviros/" + args):
            fname = self.agent.data_path + "enviros/" + args
        else: print("Error: Cannot Find Enviro Config: ", args); return

        # Load Enviro
        self.agent.set_enviro(fname)
        self.log.append("loadenviro " + args)

    def complete_loadenviro(self, text, line, begidx, endidx):
        enviro_files = [i for i in sorted(os.listdir(self.agent.data_path+"enviros"))
                        if i.startswith(text) and i.endswith('.cfg')]
        enviro_files += [i for i in sorted(os.listdir(os.getcwd()))
                        if i.startswith(text) and i.endswith('.cfg')]
        return enviro_files

    def do_test(self, args):
        '''Command runs test on agent with set routine on set Doom configuration.
        \nNote: Agent and Enviro must be loaded before running test.
        \nOptions:
        \n-v,--visualize\t
        \n-f,--full_ui\tRenders full Doom UI instead of UI limited by agent.
        \n-n,--nb_tests\t<number of tests on set Doom configuration>'''

        # Check for Agent and Enviro
        flag = False
        if not self.agent.agent_model: print("Error: No Agent Loaded."); flag = True
        if not self.agent.enviro_config: print("Error: No Enviro Loaded."); flag = True
        if flag: return

        nb_tests = 1
        visualize = False
        full_ui = False
        replay = None
        text_only = False

        # Get Options
        opts, argss = getopt.getopt(args.split(), "vtfn:r:", ['visualize', 'text_only','full_ui', 'nb_tests=', 'replay:'])
        for opt, arg in opts:
            if opt in ('-n', '--nb_tests'): nb_tests = int(arg)
            elif opt in ('-v', '--visualize'): visualize = True
            elif opt in ('-f', '--full_ui'): full_ui = True
            elif opt in ('-r', '--replay'): replay = arg
            elif opt in ('-t', '--text_only'): text_only = True

        # Run Test
        for i in range(nb_tests):
            if not replay: replay = self.agent.data_path + "test.lmp"
            self.agent.test(save_replay=replay, verbose=True, visualization=visualize)
            if not text_only: self.agent.replay(replay, doom_like=full_ui)
        self.log.append("test " + args)

    def do_replay(self, args):
        '''Command runs .lmp replay files.
        \nNote: Correct Enviro must be set before running.
        \nOptions:
        \n-f,--full_ui\tRenders full Doom UI instead of UI limited by agent.'''

        full_ui = False

        # Get Options
        opts, argss = getopt.getopt(args.split(), "f:", ['full_ui'])
        for opt, arg in opts:
            if opt in ('-f', '--full_ui'): full_ui = True
        argss = argss[0]

        # Find Replay File
        if os.path.isfile(argss): fname = argss
        elif os.path.isfile(self.agent.data_path + argss):
            fname = self.agent.data_path + argss
        else: print("Error: Cannot Find Replay: ", argss); return

        # Run Replay
        try:
            self.agent.replay(fname, doom_like=full_ui)
            self.log.append("replay " + args)
        except: print("Error: Could Not Replay: Set Correct Enviro.")

    def complete_replay(self, text, line, begidx, endidx):
        replay_files = [i for i in sorted(os.listdir(self.agent.data_path))
                        if i.startswith(text) and i.endswith('.lmp')]
        replay_files += [i for i in sorted(os.listdir(os.getcwd()))
                        if i.startswith(text) and i.endswith('.lmp')]
        return replay_files

    def do_train(self, args):
        '''Command runs train agent with set routine on set Doom configuration.
        \nNote: Routine and Doom config must be set before running test.
        \nOptions:
        \n-s,--nb_steps\t<number of steps per epoch>
        \n-e,--nb_epochs\t<number of epochs>'''

        # Check for Agent and Enviro
        flag = False
        if not self.agent.agent_model: print("Error: No Agent Loaded."); flag = True
        if not self.agent.enviro_config: print("Error: No Enviro Loaded."); flag = True
        if flag: return

        epochs = 100
        steps = 1000
        memory_size = 1000
        batch_size = 40
        gamma = 0.9
        target_update = 100
        nb_tests = 50

        # Get Options
        opts, argss = getopt.getopt(args.split(), "s:e:t:m:b:", ['nb_steps=', 'nb_epochs=',
        'nb_tests=', 'batch_size=', 'memory_size='])
        for opt, arg in opts:
            if opt in ('-s', '--nb_steps'):
                steps = int(arg)
            elif opt in ('-e', '--nb_epochs'):
                epochs = int(arg)
            elif opt in ('-t', '--nb_tests'):
                nb_tests = int(arg)
            elif opt in ('-m', '--memory_size'):
                memory_size = int(arg)
            elif opt in ('-b', '--batch_size'):
                batch_size = int(arg)

        # Train Agent
        self.agent.train(epochs, steps, memory_size, batch_size, gamma, target_update, nb_tests)
        self.log.append("train " + args)

    def do_exportscript(self, args):
        '''Command exports session as DeepDoom-DE script.'''

        # Write Log to File
        if len(args) == 0: print("Error: No File Name Defined."); return
        if not args.endswith('.dds'): script_file = args + '.dds'
        else: script_file = args
        with open(script_file, 'w') as f:
            f.write('# {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) + '\n')
            for i in range(len(self.log)):
                x = str(i+1) + " " + self.log[i] + '\n'; f.write(x)

    def do_exit(self, args):
        '''Command exits DeepDoom-DE.'''
        print("Exiting DeepDoom-DE...")
        raise SystemExit

    def help_tutorial(self):
        print('Getting Started:')
        tutorial =''' DeepDoom-DE provides researchers to define and test agents
        for Doom.'''
        print(tutorial)

    def emptyline(self): pass
