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
version = '0.0.0'

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
    install_requires=['numpy>=1.24.0', 'matplotlib>=3.6.2',
                      'pandas>=1.5.2', 'seaborn>=0.12.1',
                      'gymnasium==0.28.1', 'stable-baselines3>=1.6.2',
                      'pyzmq>=24.0.1', 'pyyaml>=5.4.1', 'tensorboard>=2.7.0', 'dvc', 'click',
                      'scipy', 'pymc>=4.0.1', 'sb3-contrib', 'ruff', 'pytest'
                      ],
    extras_require={
        "tests": [
            # Run tests and coverage
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-xdist",
            # Type check
            "pytype",
            "mypy",
            # Lint code (flake8 replacement)
            "ruff",
            # Sort imports
            "isort>=5.0",
            # Reformat
            "black",
        ],
        "docs": [
            "sphinx",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            # For spelling
            "sphinxcontrib.spelling",
            # Type hints support
            "sphinx-autodoc-typehints",
            # Copy button for code snippets
            "sphinx_copybutton",
        ]
    },
    python_requires=">=3.8",
    classifiers=['Operating System :: ubuntu',
                 'Programming Language :: Python :: 3',
                 ],
    platforms=['ALL'],
)
