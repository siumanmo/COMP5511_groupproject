#!/bin/bash
# Force clean installation of NumPy 2.0.2
pip uninstall -y numpy
pip install numpy==2.0.2 --no-cache-dir --force-reinstall