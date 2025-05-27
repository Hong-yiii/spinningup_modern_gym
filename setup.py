from os.path import join, dirname, realpath
from setuptools import setup, find_packages
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 8, \
    "The Spinning Up repo is designed to work with Python 3.8 and greater." \
    + "Please install it before proceeding."

with open(join("spinup", "version.py")) as version_file:
    exec(version_file.read())

setup(
    name='spinup',
    packages=find_packages(),
    version=__version__,
    install_requires=[
        # Core numerical and ML dependencies
        'numpy>=1.19.0',
        'scipy>=1.5.0', 
        'torch>=1.9.0',
        
        # Environment dependencies
        'gymnasium>=0.26.0',  # Modern replacement for gym
        
        # Parallel computing dependencies
        'mpi4py>=3.0.0',  # Required for distributed training
        
        # Utility dependencies (used throughout the codebase)
        'joblib>=1.0.0',  # Used in logx.py and test_policy.py
        'cloudpickle>=1.6.0',  # Used in run_utils.py  
        'psutil>=5.7.0',  # Used in run_utils.py
        'tqdm>=4.50.0',  # Used in run_utils.py
        
        # Plotting dependencies (used in utils/plot.py)
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0', 
        'pandas>=1.3.0',  # Used with seaborn in plot.py
    ],
    extras_require={
        # MPI support (optional, can cause installation issues)
        'mpi': ['mpi4py'],
        
        # Atari environments (optional, can cause installation issues)
        'atari': ['gymnasium[atari]'],
        
        # All extras
        'all': ['mpi4py', 'gymnasium[atari]'],
    },
    description="Teaching tools for introducing people to deep RL (PyTorch only, modernized).",
    author="Joshua Achiam",
    python_requires='>=3.8',
)
