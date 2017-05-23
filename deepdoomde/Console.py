#!/usr/bin/env python3
'''
Console.py
Author: Rafael Zamora
Updated: 05/16/17

'''

# Import Packages
import os, sys, getopt
from cmd import Cmd

class Console(Cmd):
    """
    Class runs DeepDoom console used to run agent training sessions and tests.

    """

    def __init__(self):
        '''
        '''
        super().__init__()

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

        self.prompt = '>>> '

        from deepdoomde.Agent import DoomAgent
        self.agent = DoomAgent()

    def do_loadagent(self, args):
        '''Command loads agent from specified .atg file.'''
        fname = self.agent.data_path + "agents/" + args
        if os.path.isfile(fname): self.agent.load_agent(args)
        else: print("Error: Cannot Find Agent Config ", args)

    def complete_loadagent(self, text, line, begidx, endidx):
        agent_files = [i for i in sorted(os.listdir(self.agent.data_path+"agents"))
                        if i.startswith(text) and i.endswith('.agt')]
        return agent_files

    def do_agentinfo(self, args):
        '''Command displays configuration information about the currently loaded agent.'''
        if self.agent: print("Displaying Loaded Agent Info..."); self.agent.agent_info()
        else: print("Error: No Agent Loaded.")

    def do_loadenviro(self, args):
        '''Command loads environment from specified .cfg file.'''
        fname = self.agent.data_path + "enviros/" + args
        if os.path.isfile(fname): self.agent.set_enviro(args)
        else: print("Error: Cannot Find Enviro Config ", fname)

    def complete_loadenviro(self, text, line, begidx, endidx):
        enviro_files = [i for i in sorted(os.listdir(self.agent.data_path+"enviros"))
                        if i.startswith(text) and i.endswith('.cfg')]
        if "vizdoom_comp_offline.cfg" in enviro_files: enviro_files.remove("vidoom_comp_offline.cfg")
        return enviro_files

    def do_test(self, args):
        '''Command runs test on agent with set routine on set Doom configuration.
        \nNote: Routine and Doom config must be set before running test.
        \nOptions:
        \n-v,--verbose\t
        \n-f,--full_ui\tRenders full Doom UI instead of UI limited by routine.
        \n-n,--nb_tests\t<number of tests on set Doom configuration>

        '''
        if not self.agent.agent_model: print("Error: No Agent Loaded."); return
        if not self.agent.enviro_config: print("Error: No Enviro Loaded."); return
        nb_tests = 1
        verbose = False
        full_ui = False
        dia_tool = False

        opts, argss = getopt.getopt(args.split(), "vdfn:", ['verbose', 'diagnostics','full_ui', 'nb_tests='])
        for opt, arg in opts:
            if opt in ('-n', '--nb_tests'):
                nb_tests = int(arg)
            elif opt in ('-v', '--verbose'):
                verbose = True
            elif opt in ('-d', '--diagnostics'):
                dia_tool = True
            elif opt in ('-f', 'full_ui'):
                full_ui = True

        # Run Tests
        for i in range(nb_tests):
            replay = "test.lmp"
            self.agent.test(save_replay=replay, verbose=True, tool=dia_tool)
            self.agent.replay(replay, verbose=verbose, doom_like=full_ui)

    def do_train(self, args):
        '''Command runs train agent with set routine on set Doom configuration.
        \nNote: Routine and Doom config must be set before running test.
        \nOptions:
        \n-s,--nb_steps\t<number of steps per epoch>
        \n-e,--nb_epochs\t<number of epochs>

        '''
        if not self.agent.agent_model: print("Error: No Agent Loaded."); return
        if not self.agent.enviro_config: print("Error: No Enviro Loaded."); return
        epochs = 100
        steps = 1000
        memory_size = 1000
        batch_size = 40
        gamma = 0.9
        target_update = 100
        nb_tests = 50

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

        self.agent.train(epochs, steps, memory_size, batch_size, gamma, target_update, nb_tests)


    def do_exit(self, args):
        '''Command exits DeepDoom environment.'''
        print("Exiting DeepDoom-DE...")
        raise SystemExit

    def help_tutorial(self):
        print('Getting Started:')
        print('a good place for a tutorial')

    def emptyline(self): pass
