import os
from setuptools import setup, find_packages

setup(
    name="balltrack",
    version="0.1",
    # packages=find_packages(),

    # dependency
    install_requires=["opencv-contrib-python>=4.0","imutils>0.5"],

    packages=find_packages(),


    # metadata
    author="kaiyao",
    author_email="kqiao@connect.ust.hk",
    license="BSD",
    url="NONE",

)
