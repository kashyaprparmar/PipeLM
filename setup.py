import os
from setuptools import setup, find_packages

# Read the requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read the README for the long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="pipelm",
    version="0.1.0",
    description="A lightweight API server and CLI for running LLM models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kashyap Parmar",
    author_email="kashyaprparmar@gmail.com",
    url="https://github.com/yourusername/pipelm",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "pipelm=pipelm.cli:main",
        ],
    },
)