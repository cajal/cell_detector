#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

long_description = "Datajoint schemata and analysis code for the AOD cell segmentation."


setup(
    name='aod_cells',
    version='0.1.0.dev1',
    description="A collection of datajoint schemata.",
    long_description=long_description,
    author='Fabian Sinz',
    author_email='sinz@bcm.edu',
    license="Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License",
    url='https://github.com/cajal/cell_detector',
    keywords='cell segmentation',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['numpy', 'h5py'],
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License',
        'Topic :: Database :: Front-Ends',
    ],
    scripts=[
        'scripts/cell_finder',
        'scripts/label_cells.py',
        'scripts/predict_stack.py',
    ]
)
