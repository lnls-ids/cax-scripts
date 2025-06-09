#!/usr/bin/env python-sirius

from setuptools import setup, find_packages
import os

version_file = os.path.join(os.path.dirname(__file__), 'VERSION')
with open(version_file, 'r') as _f:
    __version__ = _f.read().strip()

setup(
    name='caxscripts',
    version=__version__,
    author='lnls-ids',
    description='Scripts for beam analysis',
    url='https://github.com/lnls-ids/cax-scripts',
    download_url='https://github.com/lnls-ids/cax-scripts',
    license='MIT License',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering'
    ],
    packages=find_packages(),
    package_data={
        'caxscripts': ['VERSION'],
    },
    include_package_data=True,
    zip_safe=False)