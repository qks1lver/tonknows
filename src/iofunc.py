#!/usr/bin/env python3

"""
iofunc.py
"""



# Import
import pickle
import code

# Functions
def open_pkl(p_file):

    with open(p_file, 'rb') as file:
        return pickle.load(file)

def save_pkl(p_file, content):

    with open(p_file, 'wb') as f:
        pickle.dump(content, f)

    return

def interact(var_desc=None, local=None):

    print('\n__/ Interactive session \_____________________________________')

    if not local:
        print('\n  ** Nothing in interactive workspace **')
        local = globals()

    elif var_desc:
        print('\n     Variables:')
        for v, d in var_desc.items():
            print('        {:>8}  {:<10}'.format(v, d))

    print('______                    ___________')
    print('      \ Ctrl + D to exit /\n')
    code.interact(local=local)

    return
