#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os.path import exists
from setuptools import setup, find_packages

author = 'Serhii'
email = 'serhii.aif@mpl.mpg.de'
description = 'A package for reinforcement learning in adaptive therapy'
name = 'physilearning'
year = '2023'
url = ''
version = '0.3.0'

setup(
    name=name,
    author=author,
    author_email=email,
    url=url,
    version=version,
    packages=find_packages('physilearning'),
    package_dir={'': 'src'},
    license='None',
    description=description,
    long_description=open('README.md').read() if exists('README.md') else '',
    long_description_content_type="text/markdown",
    install_requires=['numpy==1.23.5', 'matplotlib>=3.6.2',
                      'pandas>=1.5.2', 'tables>=3.5.0', 'seaborn>=0.12.1',
                      'gymnasium', 'stable-baselines3[extra]>=2.0.0',
                      'pyzmq>=24.0.1', 'pyyaml>=5.4.1', 'tensorboard>=2.7.0', 'dvc', 'click',
                      'scipy>=1.10.1', 'pymc>=4.0.1', 'sb3-contrib', 'ruff', 'pytest', 'pytest-cov',
                      'sphinx<7.0.0', 'sphinx_rtd_theme', 'sphinx-autodoc-typehints',
                      'mpl-scatter-density', 'arviz', 'pytensor'
                      ],
    python_requires=">=3.10",
    classifiers=['Operating System :: ubuntu',
                 'Programming Language :: Python :: 3',
                 ],
    platforms=['Linux'],
)
