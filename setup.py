from setuptools import setup, find_packages

setup(
    name="find_robust_seed",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "tqdm"
    ],
    extras_require={
        'test': [
            'pytest',
            'pytest-cov'
        ]
    },
    author="Luca Zavarella",
    description="A package for finding robust random seeds for dataset splitting",
    python_requires=">=3.7",
    test_suite='tests',
)
