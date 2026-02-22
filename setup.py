from setuptools import setup, find_packages

setup(
    name="fedctx_iot",
    version="1.0.0",
    author="Ali Mansour Al-madani",
    author_email="ali.m.almadani1992@gmail.com",
    description=(
        "FedCTX-IoT: A Privacy-Preserving Federated CNN-Transformer "
        "Framework with Dual-Pathway Explainability for IoT Intrusion Detection"
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/<your-username>/fedctx-iot",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "shap>=0.42.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "opacus>=1.4.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
)
