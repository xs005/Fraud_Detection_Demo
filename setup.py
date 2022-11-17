#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='Chris_Sun_Fraud_Detection_Demo',
    setup_requires=['pbr>=1.9', 'setuptools>=17.1'],
    pbr=True,
    packages=find_packages(),
    python_requires='>=3.10'
)