#!/usr/bin/env python3

__author__ = 'Rafael Zamora, rz4@hood.edu'

from setuptools import setup, find_packages

setup(
    name="DeepDoom-DE",
    version="0.1.0",
    description="Deep Reinforcement Learning Development Environment Powered By ViZDoom 1.1.1.",
    license="MIT",
    keywords="Doom Deep Learning",
    packages=find_packages(exclude=["DeepDoom-DE"]),
    package_data={'agent_config': ['agent_config.cfg'], 'deepdoomwad':['deepdoom.wad']},
    include_package_data=True,
    install_requires = ["vizdoom", "keras", "tensorflow", "matplotlib", "tqdm", "keras-vis"],
)