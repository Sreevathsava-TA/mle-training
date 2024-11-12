import codecs
import os

from setuptools import find_packages, setup

setup(
    name="mle_training",  # Replace with your project name
    version="0.0.1",  # Increment as necessary
    description="A machine learning project for housing data analysis.",
    author="Sreevathsava",  # Replace with your name or your organization's name
    author_email="sreevathsava.pak@tigeranalytics.com",  # Replace with your contact email
    url="https://github.com/Sreevathsava-TA/mle-training",  # Replace with your project's GitHub repo or URL
    packages=find_packages("src"),  # Includes all packages under the 'src' directory
    package_dir={"": "src"},  # Tells setuptools to look for packages in 'src'
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "pyyaml",
        "black",
        "isort",
        "flake8"
    ],
    extras_require={
        "dev": [
            "pytest",
            "sphinx",
            "black",
            # Include any additional dependencies for development and testing
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "fetch-data=project_name:fetch_housing_data",
            "train-model=project_name.model_training:train_model",
            "evaluate-model=project_name.model_training:evaluate_model",
            # Add other script entry points if needed
        ],
    },
)
