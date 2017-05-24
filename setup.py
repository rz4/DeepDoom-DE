#!/usr/bin/env python3

__author__ = 'Rafael Zamora, rz4@hood.edu'

from setuptools import setup, find_packages

setup(
    name="DeepDoom-DE",
    version="0.1.0",
    description="Deep Reinforcement Learning Development Environment Powered By ViZDoom 1.1.1. and Keras 2.0",
    license="MIT",
    keywords="Doom Deep Reinforcement Learning",
    packages=find_packages(exclude=["agents","enviros","docker"]),
    package_data={'deepdoomde':['agent_config.cfg','deepdoom.wad']},
    include_package_data=True,
    install_requires = ["numpy","keras", "tensorflow", "matplotlib", "tqdm", "opencv-python", "keras-vis", "wget", "vizdoom"],
)
