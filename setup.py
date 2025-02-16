from setuptools import setup, find_packages

setup(
    name="grnsuite",
    version="0.1.0",
    author="Rachel H. Parkinson",
    description="A package for analyzing insect taste electrophysiology recordings",
    packages=find_packages(),
    install_requires=[
        "numpy", "scipy", "matplotlib", "pandas", "scikit-learn", "click", "pytest", "snakemake"
    ],
    entry_points={
        "console_scripts": [
            "grnsuite=grnsuite.cli:cli"
        ]
    },
)