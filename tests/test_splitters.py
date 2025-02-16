# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, GroupShuffleSplit
from find_robust_seed import find_best_split_seed

# %%
# Create sample data
np.random.seed(42)
n_samples = 1000

# Features
X = pd.DataFrame({
    'numeric1': np.random.normal(0, 1, n_samples),
    'numeric2': np.random.normal(0, 1, n_samples),
    'category1': np.random.choice(['A', 'B', 'C'], n_samples),
    'category2': np.random.choice(['X', 'Y'], n_samples)
})

# Target (both numeric and categorical cases)
y_numeric = pd.Series(np.random.normal(0, 1, n_samples), name='target_numeric')
y_categorical = pd.Series(np.random.choice(['0', '1'], n_samples), name='target_categorical')

# Groups for GroupShuffleSplit
groups = np.random.choice(['g1', 'g2', 'g3', 'g4'], n_samples)

# Define columns
categorical_columns = ['category1', 'category2']
numerical_columns = ['numeric1', 'numeric2']

# %%
print("Testing different splitters...")

# Test 1: ShuffleSplit
print("\n1. Testing ShuffleSplit with numeric target")
splitter = ShuffleSplit(n_splits=5, test_size=0.2)
result = find_best_split_seed(
    X=X,
    y=y_numeric,
    splitter=splitter,
    categorical_columns=categorical_columns,
    numerical_columns=numerical_columns,
    n_samples=10,  # Small number for quick testing
    verbose=True
)
print(f"Best seed found: {result['best_seed']}")

# Test 2: StratifiedShuffleSplit with categorical target
print("\n2. Testing StratifiedShuffleSplit with categorical target")
splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
result = find_best_split_seed(
    X=X,
    y=y_categorical,
    splitter=splitter,
    categorical_columns=categorical_columns,
    numerical_columns=numerical_columns,
    n_samples=10,  # Small number for quick testing
    verbose=True
)
print(f"Best seed found: {result['best_seed']}")

# Test 3: GroupShuffleSplit
print("\n3. Testing GroupShuffleSplit")
splitter = GroupShuffleSplit(n_splits=5, test_size=0.2)
result = find_best_split_seed(
    X=X,
    y=y_numeric,
    splitter=splitter,
    categorical_columns=categorical_columns,
    numerical_columns=numerical_columns,
    n_samples=10,  # Small number for quick testing
    groups=groups,  # Pass the groups parameter
    verbose=True
)
print(f"Best seed found: {result['best_seed']}")

print("\nAll tests completed successfully!")

# %%
