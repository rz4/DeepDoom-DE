#!/usr/bin/env python3
'''
__init__.py
Author: Rafael Zamora
Updated: 5/22/17

'''
import os, sys, getopt, wget
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
    console.cmdloop()

def run_script(script_file):
    '''
    Method runs DeepDoom-DE script.

    '''
    if os.path.exists(script_file):
        if script_file.endswith(".dds"):
            from deepdoomde.Console import Console
            console = Console(verbose=False)
            print("\nDeepDoom-DE Running Script: " + script_file)
            with open(script_file) as f:
                i=0
                for line in f:
                    if i > 0:
                        print("\n"+line)
                        arg = line[1:]
                        l = console.precmd(arg)
                        r = console.onecmd(l)
                        r = console.postcmd(r, l)
                    i += 1
        else: print("Error: Incorrect File Type: " + script_file)
    else: print("Error: Could Not Find Script: " + script_file)

def main():
    '''
    '''
    # Make DeepDoom-DE directory
    if not os.path.isdir(os.path.expanduser('~') + '/.deepdoomde'):
        os.mkdir(os.path.expanduser('~') + '/.deepdoomde')
    if not os.path.isdir(os.path.expanduser('~') + '/.deepdoomde/agents'):
        os.mkdir(os.path.expanduser('~') + '/.deepdoomde/agents')
    if not os.path.isdir(os.path.expanduser('~') + '/.deepdoomde/enviros'):
        os.mkdir(os.path.expanduser('~') + '/.deepdoomde/enviros')
    if not os.path.isdir(os.path.expanduser('~') + '/.deepdoomde/enviros/wads'):
        os.mkdir(os.path.expanduser('~') + '/.deepdoomde/enviros/wads')

    if not os.path.exists(os.path.expanduser('~') + '/.deepdoomde/agents/rigid_turner.agt'):
        wget.download('https://raw.githubusercontent.com/rz4/DeepDoom-DE/master/agents/rigid_turner.agt', out=os.path.expanduser('~') + '/.deepdoomde/agents/', bar=None)
    if not os.path.exists(os.path.expanduser('~') + '/.deepdoomde/agents/exit_finder.agt'):
        wget.download('https://raw.githubusercontent.com/rz4/DeepDoom-DE/master/agents/exit_finder.agt', out=os.path.expanduser('~') + '/.deepdoomde/agents/', bar=None)
    if not os.path.exists(os.path.expanduser('~') + '/.deepdoomde/agents/shooter.agt'):
        wget.download('https://raw.githubusercontent.com/rz4/DeepDoom-DE/master/agents/shooter.agt', out=os.path.expanduser('~') + '/.deepdoomde/agents/', bar=None)
    if not os.path.exists(os.path.expanduser('~') + '/.deepdoomde/agents/door_opener.agt'):
        wget.download('https://raw.githubusercontent.com/rz4/DeepDoom-DE/master/agents/door_opener.agt', out=os.path.expanduser('~') + '/.deepdoomde/agents/', bar=None)
    if not os.path.exists(os.path.expanduser('~') + '/.deepdoomde/enviros/rigid_turning.cfg'):
        wget.download('https://raw.githubusercontent.com/rz4/DeepDoom-DE/master/enviros/rigid_turning.cfg', out=os.path.expanduser('~') + '/.deepdoomde/enviros/', bar=None)
    if not os.path.exists(os.path.expanduser('~') + '/.deepdoomde/enviros/exit_finding.cfg'):
        wget.download('https://raw.githubusercontent.com/rz4/DeepDoom-DE/master/enviros/exit_finding.cfg', out=os.path.expanduser('~') + '/.deepdoomde/enviros/', bar=None)
    if not os.path.exists(os.path.expanduser('~') + '/.deepdoomde/enviros/shooting.cfg'):
        wget.download('https://raw.githubusercontent.com/rz4/DeepDoom-DE/master/enviros/shooting.cfg', out=os.path.expanduser('~') + '/.deepdoomde/enviros/', bar=None)
    if not os.path.exists(os.path.expanduser('~') + '/.deepdoomde/enviros/doors.cfg'):
        wget.download('https://raw.githubusercontent.com/rz4/DeepDoom-DE/master/enviros/doors.cfg', out=os.path.expanduser('~') + '/.deepdoomde/enviros/', bar=None)
    if not os.path.exists(os.path.expanduser('~') + '/.deepdoomde/enviros/all_skills.cfg'):
        wget.download('https://raw.githubusercontent.com/rz4/DeepDoom-DE/master/enviros/all_skills.cfg', out=os.path.expanduser('~') + '/.deepdoomde/enviros/', bar=None)
    if not os.path.exists(os.path.expanduser('~') + '/.deepdoomde/enviros/rigid_turning_validation.cfg'):
        wget.download('https://raw.githubusercontent.com/rz4/DeepDoom-DE/master/enviros/rigid_turning_validation.cfg', out=os.path.expanduser('~') + '/.deepdoomde/enviros/', bar=None)
    if not os.path.exists(os.path.expanduser('~') + '/.deepdoomde/enviros/wads/rigid_turning.wad'):
        wget.download('https://raw.githubusercontent.com/rz4/DeepDoom-DE/master/enviros/wads/rigid_turning.wad', out=os.path.expanduser('~') + '/.deepdoomde/enviros/wads/', bar=None)
    if not os.path.exists(os.path.expanduser('~') + '/.deepdoomde/enviros/wads/exit_finding.wad'):
        wget.download('https://raw.githubusercontent.com/rz4/DeepDoom-DE/master/enviros/wads/exit_finding.wad', out=os.path.expanduser('~') + '/.deepdoomde/enviros/wads/', bar=None)
    if not os.path.exists(os.path.expanduser('~') + '/.deepdoomde/enviros/wads/shooting.wad'):
        wget.download('https://raw.githubusercontent.com/rz4/DeepDoom-DE/master/enviros/wads/shooting.wad', out=os.path.expanduser('~') + '/.deepdoomde/enviros/wads/', bar=None)
    if not os.path.exists(os.path.expanduser('~') + '/.deepdoomde/enviros/wads/doors.wad'):
        wget.download('https://raw.githubusercontent.com/rz4/DeepDoom-DE/master/enviros/wads/doors.wad', out=os.path.expanduser('~') + '/.deepdoomde/enviros/wads/', bar=None)
    if not os.path.exists(os.path.expanduser('~') + '/.deepdoomde/enviros/wads/all_skills.wad'):
        wget.download('https://raw.githubusercontent.com/rz4/DeepDoom-DE/master/enviros/wads/all_skills.wad', out=os.path.expanduser('~') + '/.deepdoomde/enviros/wads/', bar=None)
    if not os.path.exists(os.path.expanduser('~') + '/.deepdoomde/enviros/wads/rigid_turning_validation.wad'):
        wget.download('https://raw.githubusercontent.com/rz4/DeepDoom-DE/master/enviros/wads/rigid_turning_validation.wad', out=os.path.expanduser('~') + '/.deepdoomde/enviros/wads/', bar=None)

    argv = sys.argv[1:]

    if len(argv) == 0:
        # Run DeepDoom-DE Console
        run_console()
    else:
        try: opts, args = getopt.getopt(argv,"hcs:",['script=','console'])
        except getopt.GetoptError as e: print("Error: " + str(e)); sys.exit(2)
        for opt, arg in opts:
            if opt in ('-h'):
                # Display Help
                print("|Command\t|Argument\t|Description")
                print("-h\t\t\t\tdisplay help")
                print("-c,--console\t\t\trun console")
                print("-s,--script\t<script file>\trun script")
            elif opt in ('-c', '--console'):
                # Run DeepDoom-DE Console
                run_console()
            elif opt in ('-s', '--script'):
                # Run DeepDoom-DE Script
                run_script(arg)

    from keras import backend as K; K.clear_session()
