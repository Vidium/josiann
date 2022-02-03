#!/bin/bash

echo "-- FLAKE8 ------------------------------"
flake8 --config .tox.ini ./josiann
echo

echo "-- MYPY --------------------------------"
mypy --config-file .mypy.ini ./josiann
echo

echo "-- PYLINT ------------------------------"
pylint ./josiann
echo

echo "-- PYTEST ------------------------------"
python -m pytest -v ./tests/test_sa.py