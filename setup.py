from setuptools import setup, setuptools 
from eeg2vec._version import __version__

setup(
    name = 'eeg2vec.py',
    packages = setuptools.find_packages(),
    author = 'The Mayo Clinic Neurology AI Program',
    version = __version__
)