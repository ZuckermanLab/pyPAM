from setuptools import setup, find_packages

# Read the contents of the requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="pyPAM",
    version="0.0.1",
    description="A parallelized extenional of the affine invariant ensemble sampler, with optional mixing stages.",
    packages=find_packages(),
    install_requires=requirements,  # Set the install_requires parameter to the contents of requirements.txt
)
