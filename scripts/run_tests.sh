#!/bin/bash
python3 -m pytest --cov-config .coveragerc --cov-report html --cov-report term --cov-report xml --cov=./src/physilearning -v --color=yes -m "not expensive"