from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Package dependencies
with open("requirements.txt", encoding="utf-8") as f:
    REQUIREMENTS = [l.strip() for l in f.readlines() if l]

setup(
    name="comseg",
    version="0.0.1",
    author="Thomas Defard",
    author_email="thomas.defard@mines-paristech.fr",
    description="CNN framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tdefa/ComSeg_pkg",
    project_urls={"Bug Tracker": "https://github.com/tdefa/ComSeg_pkg/issues"},
    classifiers=[
        "Programming Language :: Python :: 3.8",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=REQUIREMENTS,
)