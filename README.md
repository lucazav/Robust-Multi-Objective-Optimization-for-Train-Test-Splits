# Robust Multi-Objective Optimization for Train-Test Splits

This repository contains a Python package designed to select the most robust random seed for train-test splits. By evaluating multiple statistical metrics simultaneously, the package ensures that the resulting training and test datasets best preserve the overall characteristics of the original data, leading to fairer model evaluations and more reliable generalization.

## Overview

When splitting datasets, even minor variations in the random seed can lead to significant differences in univariate and multivariate statistics. This package tackles the issue by:

- **Assessing Multiple Metrics:** Evaluates robust Mahalanobis distance, Wasserstein distance, Jensen–Shannon Divergence, Spearman correlation differences, Cramér's V difference, and numeric–categorical association differences.
- **Multi-Objective Optimization:** Normalizes diverse metrics and uses Pareto frontier analysis to find the optimal seed that minimizes the overall divergence from the original dataset.
- **Flexibility:** Supports any scikit-learn splitter (e.g., `ShuffleSplit`, `StratifiedShuffleSplit`, `GroupShuffleSplit`), along with optional group and stratification parameters.

The approach detailed in this repository is based on the analysis described in the article:

[Unexpected Strength of a Random Seed Change (and How to Control It): The Hidden Defect in Your Machine Learning Model](https://medium.com/data-science-collective/unexpected-strength-of-a-random-seed-change-and-how-to-control-it-the-hidden-defect-in-your-ml-0a824dea3ecf)

## Repository Structure

```
robust-multi-objective-optimization-for-train-test-splits/
├── find_robust_seed/
│   ├── __init__.py        # Exports the main functions.
│   ├── core.py            # Contains the primary implementation of find_best_split_seed and compare_split_distributions.
│   └── utils.py           # Helper functions for statistical computations and normalization.
├── tests/
│   ├── __init__.py        # Exports the main functions.
│   ├── test_splitters.py  # Contains code to test the exposed functions of the package.
├── conda.yml              # Conda environment file for easy environment setup.
├── LICENSE                # License file.
├── README.md              # This file.
└── setup.py               # Setup script for package installation.
```

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/robust-multi-objective-optimization-for-train-test-splits.git
   cd robust-multi-objective-optimization-for-train-test-splits
   ```
2. **Create and Activate the Conda Environment:**

```bash
conda env create -f conda.yml
conda activate ml_tools
```

3. **Install the Package in Editable Mode:**
   Installing in editable mode via `pip install -e .` ensures that any changes you make to the source code are immediately available in your environment without the need for reinstallation.

```bash
pip install -e .
```

## Usage

The package provides two main functions:

* find_best_split_seed(): Evaluates a range of random seeds using multi-objective optimization to select the seed that best preserves the dataset's statistical structure in the train-test split.
* compare_split_distributions(): Generates a detailed comparison of key summary statistics between the original dataset and the splits obtained using the selected seed.

A typical workflow is demonstrated in the `fish_market_demo.py` script you can find [here](https://gist.github.com/lucazav/cbbb294e075a86ca1e4e6eab73605fc8 "fish_market_demo.py Gist").

## Key Features

* **Robust Statistical Metrics:**
  Evaluate the alignment of numeric distributions (via *robust Mahalanobis* and *Wasserstein* distances) and categorical distributions (via *Jensen–Shannon Divergence* and *Cramér's V*).
* **Preservation of Data Structure:**
  Ensure that the inter-feature relationships (e.g., *Spearman correlation*, *numeric–categorical associations*) remain consistent between the training and test splits.
* **Flexible Splitter Support:**
  Easily integrate any scikit-learn splitter by passing the splitter object along with optional parameters like `groups` for group-based splits.

Here's the updated **Contributing** section with your request included:

## Contributing

Contributions are welcome! If you have suggestions, bug reports, or improvements, please open an issue or submit a pull request on GitHub.

This project is an example of how you can implement advanced ideas using AI-powered tools. As I am not an expert Pythonista, I leveraged **[Cline](https://github.com/cline/cline "An AI assistant that can use your CLI aNd Editor")**, the VS Code extension for AI-driven coding assistance, to structure and refine this implementation. Feel free to improve the codebase or suggest optimizations!
