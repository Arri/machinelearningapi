## Setup file use:
#  python setup.py install
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name="machinelearningapi", 
    version="0.0.1",
    author="Arasch U Lagies",
    author_email="arasch.lagies@axiado.com",
    description="Numpy based Machine Learning API",
    long_description=long_description,
    long_description_content_type="",
	url="http://10.4.1.42/software/machinelearningapi",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)