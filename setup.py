
# -*- coding: utf-8 -*-
# 2021 Machine Learning Course Alfatraining
# Author: J. Caron
#

import setuptools

print(setuptools.find_packages())

setuptools.setup(
    name="mlalfa",
    version="0.0.1-dev",
    author="Jan Caron",
    author_email="jan.caron@web.de",
    description="machine learning models from alfatraining course",
    long_description='',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
