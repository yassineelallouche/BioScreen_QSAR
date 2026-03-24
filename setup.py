"""
setup.py — BioScreen-QSAR
Install with: pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="bioscreen-qsar",
    version="1.0.0",
    author="BioScreen-QSAR Project",
    description="A modular low-code QSAR framework for antimicrobial activity prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/BioScreen-QSAR",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "bioscreen-curate  = scripts.data_curation:main",
            "bioscreen-dedupe  = scripts.duplicate_handling:main",
            "bioscreen-fp      = scripts.descriptors_ecfp:main",
            "bioscreen-train   = scripts.train_classification_models:main",
            "bioscreen-screen  = scripts.virtual_screening:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    keywords="QSAR, drug discovery, antimicrobial, machine learning, cheminformatics, RDKit",
)
