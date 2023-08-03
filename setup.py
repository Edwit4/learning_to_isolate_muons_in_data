# setup.py
#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="src",
    version="0.0.1",
    description="Muon classification on unlabeled CMS data",
    author="Ed Witkowski",
    author_email="edwit4@gmail.com",
    url="https://github.com/Edwit4/cms_classification",
    install_requires=["pytorch-lightning", 
                      "hep-ml",
                      "average-decision-ordering",
                      "torch",
                      "numpy", 
                      "scipy", 
                      "lmdb", 
                      "matplotlib",
                      "scikit-learn", 
                      "pandas"],
    packages=find_packages(exclude=['training', 'data']),
)