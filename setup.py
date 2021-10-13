import setuptools

long_description = """
This package aims at the automation of useful steps that need to be taken in every single Machine Learning and Data Science project.
It has some functions to easily and readily plot data visualizations - not only to study the data but also to evaluate a given model's performance.
Also, some custom preprocessors are made available in order to be integrated into scikit-learn pipelines.
"""
with open("requirements.txt", "r") as file:
    required_packages = [package.strip() for package in file.readlines()]

setuptools.setup(
    name="light-ml",
    version="0.0.1",
    author="Gabriel Fernandes Luz",
    author_email="gfluz94@gmail.com",
    description="Package to automate data science and machine learning pipelines.",
    long_description=long_description,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=required_packages
)