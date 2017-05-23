#!/usr/bin/env python3
'''
__init__.py
Author: Rafael Zamora
Updated: 5/22/17

'''
import os, sys, getopt
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def run_console():
    '''
    Method runs DeepDoom-DE console.

    '''
    # Import Packages
    from deepdoomde.Console import Console

    # Initiate and Run Console
    console = Console()
    console.cmdloop('\nWelcome to DeepDoom-DE! For more information use the \'help\' command.')

def run_script():
    '''
    Method runs DeepDoom-De script.

    '''
    pass

def main():
    '''
    '''
    if not os.path.isdir('DeepDoom-DE'): os.mkdir('DeepDoom-DE')
    if not os.path.isdir('DeepDoom-DE/agents'): os.mkdir('DeepDoom-DE/agents')
    if not os.path.isdir('DeepDoom-DE/enviros'): os.mkdir('DeepDoom-DE/enviros')
    if not os.path.isdir('DeepDoom-DE/enviros/wads'): os.mkdir('DeepDoom-DE/enviros/wads')
    if not os.path.isdir('DeepDoom-DE/results'): os.mkdir('DeepDoom-DE/results')
    if not os.path.isdir('DeepDoom-DE/results/graphs'): os.mkdir('DeepDoom-DE/results/graphs')
    if not os.path.isdir('DeepDoom-DE/results/replays'): os.mkdir('DeepDoom-DE/results/replays')

    argv = sys.argv[1:]

    if len(argv) == 0:
        # Run DeepDoom-DE Console
        run_console()
    else:
        # Run DeepDoom-DE Script
        try:
            opts, args = getopt.getopt(argv,"",[])
        except getopt.GetoptError:
             print()
             sys.exit(2)
        for opt,arg in opts:
            if opt in ():
                print("hello")
