#!/bin/bash

# Update package manager (optional, but recommended)
sudo apt update

# Install Python and pip (if not already installed)
sudo apt install python3 python3-pip

# Install project-specific Python packages using pip
pip install -r requirements.txt

# Install other system-level dependencies if needed
# For example:
# sudo apt install some-package

# Additional setup or configuration steps if necessary
# For example:
# python manage.py migrate
