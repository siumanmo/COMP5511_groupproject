#!/bin/bash
# Remove existing installations
pip uninstall -y numpy pandas scikit-learn

# Install exact versions with no caching
pip install --no-cache-dir numpy==1.21.6 pandas==1.3.5 scikit-learn==1.0.2

# Install remaining requirements
pip install -r requirements.txt
