from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="GEN-AI Project",
    version="0.1",
    author="Jeffrey",
    author_email="vgauss23@outlook.com",
    packages=find_packages(),
    install_requires=requirements,
)
