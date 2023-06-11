"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

# For relative imports to work in Python 3.6+
#then we can replace relative imports with regular imports in other modules in this package - like models.py, data.py etc
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from . import explore, train, train_multigpu, train_multigpu_2T
