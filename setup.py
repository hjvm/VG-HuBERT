"""
Setup script for VG-HuBERT package.

This enables pip installation and HuggingFace Hub integration.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vg-hubert",
    version="1.0.0",
    author="Puyuan Peng, David Harwath (original); YOUR_NAME (fork maintainer)",
    author_email="YOUR_EMAIL",
    description="VG-HuBERT: Simplified interface for speech segmentation (no PyTorch patching required)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/VG-HuBERT",
    project_urls={
        "Original Word Discovery": "https://github.com/jasonppy/word-discovery",
        "Original Syllable Discovery": "https://github.com/jasonppy/syllable-discovery",
        "Fork Parent": "https://github.com/human-ai-lab/VG-HuBERT",
        "Interface Inspiration (Sylber)": "https://github.com/Berkeley-Speech-Group/sylber",
        "HuggingFace Model": "https://huggingface.co/YOUR_USERNAME/vg-hubert",
    },
    packages=find_packages(include=["vg_hubert", "vg_hubert.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",  # Required for native eager attention support
        "transformers>=4.20.0",
        "huggingface-hub>=0.10.0",
        "numpy>=1.20.0",
        "soundfile>=0.10.0",
        "scipy>=1.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
        "training": [
            "fairseq>=0.10.0",  # Only needed for training, not inference
            "Pillow>=9.0.0",
            "matplotlib>=3.5.0",
            "scikit-learn>=1.1.0",
            "seaborn>=0.11.0",
            "tqdm>=4.60.0",
            # Note: apex must be installed separately from https://github.com/NVIDIA/apex
        ],
    },
    include_package_data=True,
    package_data={
        "vg_hubert": ["*.yaml", "*.json"],
    },
    license="BSD-3-Clause",
    keywords="speech audio segmentation syllables self-supervised hubert vg-hubert",
)
