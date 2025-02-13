# %%
import numpy as np
import pandas as pd
import janitor
import random
import time
import logging
import json
import warnings
import os
import concurrent.futures

from datetime import datetime
from numpy.linalg import norm
from scipy.stats import spearmanr
from sklearn.model_selection import ShuffleSplit
from sklearn.covariance import MinCovDet
from concurrent.futures import ThreadPoolExecutor

# Optional: tqdm for progress bars (if available)
try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False


# %%
# Configure logging to output timestamp, log level, and message.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


###############################################################################
# Module-Level Helper Functions
###############################################################################

def compare_split_distributions(X, y, best_seed, test_size=0.2, 
                               categorical_columns=None, numerical_columns=None):
    """
    Compare key distribution metrics between the full dataset, training set, and test set
    for the best found seed. Returns a DataFrame with metrics for easy comparison.
    
    Parameters:
    -----------
    X : pandas DataFrame
        The feature matrix
    y : pandas Series or numpy array
        The target variable
    best_seed : int
        Random seed for reproducible splits
    test_size : float, default=0.2
        Proportion of dataset to include in the test split
    categorical_columns : list of str, optional
        Names of categorical columns
    numerical_columns : list of str, optional
        Names of numerical columns
        
    Returns:
    --------
    pandas DataFrame
        A DataFrame containing distribution metrics for comparison
    """
    try:
        # Input validation
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if not isinstance(test_size, float) or not 0 < test_size < 1:
            raise ValueError("test_size must be a float between 0 and 1")
        if isinstance(y, pd.DataFrame):
            target_name = y.columns[0]
            y = y.squeeze()
            
        # Determine column types if not provided
        if categorical_columns is None and numerical_columns is None:
            numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_columns = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
        elif categorical_columns is None:
            categorical_columns = [col for col in X.columns if col not in numerical_columns]
        elif numerical_columns is None:
            numerical_columns = [col for col in X.columns if col not in categorical_columns]
            
        # Create the split
        cv = ShuffleSplit(n_splits=1, test_size=test_size, random_state=best_seed)
        train_idx, test_idx = next(cv.split(X))
        
        X_tr = X.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_tr = y.iloc[train_idx] if isinstance(y, (pd.Series, pd.DataFrame)) else y[train_idx]
        y_te = y.iloc[test_idx] if isinstance(y, (pd.Series, pd.DataFrame)) else y[test_idx]
        
        # Prepare results dictionary
        results = {
            'Metric': [],
            'Full Dataset': [],
            'Training Set': [],
            'Test Set': []
        }
        
        # Helper function to add metrics
        def add_metric(name, full_val, train_val, test_val):
            results['Metric'].append(name)
            results['Full Dataset'].append(full_val)
            results['Training Set'].append(train_val)
            results['Test Set'].append(test_val)
        
        # 1. Numeric feature distributions and correlations
        if numerical_columns:
            # Distribution statistics for each numeric feature
            for col in numerical_columns:
                # Basic statistics
                add_metric(f"Mean ({col})", 
                          X[col].mean(),
                          X_tr[col].mean(),
                          X_te[col].mean())
                
                add_metric(f"Std ({col})",
                          X[col].std(),
                          X_tr[col].std(),
                          X_te[col].std())
                
                # Quartiles and median
                add_metric(f"25th Percentile ({col})",
                          X[col].quantile(0.25),
                          X_tr[col].quantile(0.25),
                          X_te[col].quantile(0.25))
                
                add_metric(f"Median ({col})",
                          X[col].median(),
                          X_tr[col].median(),
                          X_te[col].median())
                
                add_metric(f"75th Percentile ({col})",
                          X[col].quantile(0.75),
                          X_tr[col].quantile(0.75),
                          X_te[col].quantile(0.75))
                
                # Additional statistics
                add_metric(f"Skewness ({col})",
                          X[col].skew(),
                          X_tr[col].skew(),
                          X_te[col].skew())
                
                add_metric(f"Kurtosis ({col})",
                          X[col].kurtosis(),
                          X_tr[col].kurtosis(),
                          X_te[col].kurtosis())
                
                add_metric(f"Min ({col})",
                          X[col].min(),
                          X_tr[col].min(),
                          X_te[col].min())
                
                add_metric(f"Max ({col})",
                          X[col].max(),
                          X_tr[col].max(),
                          X_te[col].max())
                
                # Wasserstein distance
                full_sorted = np.sort(X[col].values)
                train_sorted = np.sort(X_tr[col].values)
                test_sorted = np.sort(X_te[col].values)
                
                # Compute CDFs
                full_cdf = np.linspace(0, 1, len(full_sorted))
                train_cdf = np.linspace(0, 1, len(train_sorted))
                test_cdf = np.linspace(0, 1, len(test_sorted))
                
                # Interpolate to common grid
                grid = np.unique(np.concatenate([full_sorted, train_sorted, test_sorted]))
                full_interp = np.interp(grid, full_sorted, full_cdf)
                train_interp = np.interp(grid, train_sorted, train_cdf)
                test_interp = np.interp(grid, test_sorted, test_cdf)
                
                # Compute Wasserstein distances
                train_wasserstein = np.trapezoid(np.abs(train_interp - full_interp), grid)
                test_wasserstein = np.trapezoid(np.abs(test_interp - full_interp), grid)
                
                add_metric(f"Wasserstein Distance ({col})",
                          0.0,  # Reference distance to full dataset is 0
                          train_wasserstein,
                          test_wasserstein)
            
            # Spearman correlations between numeric features
            if len(numerical_columns) > 1:
                full_corr = spearmanr(X[numerical_columns]).correlation
                train_corr = spearmanr(X_tr[numerical_columns]).correlation
                test_corr = spearmanr(X_te[numerical_columns]).correlation
                
                for i, col1 in enumerate(numerical_columns):
                    for j, col2 in enumerate(numerical_columns):
                        if i < j:  # Only show upper triangle
                            add_metric(f"Spearman Correlation ({col1} vs {col2})",
                                     full_corr[i, j],
                                     train_corr[i, j],
                                     test_corr[i, j])
        
        # 2. Categorical distributions, correlations, and associations
        if categorical_columns:
            # Distribution statistics for each categorical feature
            for col in categorical_columns:
                # Convert to pandas Series and get value counts
                full_dist = pd.Series(X[col]).value_counts(normalize=True)
                train_dist = pd.Series(X_tr[col]).value_counts(normalize=True)
                test_dist = pd.Series(X_te[col]).value_counts(normalize=True)
                
                # Get counts for cardinality metrics
                full_counts = pd.Series(X[col]).value_counts()
                train_counts = pd.Series(X_tr[col]).value_counts()
                test_counts = pd.Series(X_te[col]).value_counts()
                
                # Add cardinality metrics
                add_metric(f"Unique Values Count ({col})",
                          len(full_counts),
                          len(train_counts),
                          len(test_counts))
                
                add_metric(f"Most Frequent Value Count ({col})",
                          full_counts.iloc[0],
                          train_counts.iloc[0] if len(train_counts) > 0 else 0,
                          test_counts.iloc[0] if len(test_counts) > 0 else 0)
                
                # Convert indices to lists and get unique categories
                all_cats = pd.Index(set(list(full_dist.index) + 
                                      list(train_dist.index) + 
                                      list(test_dist.index)))
                
                # Reindex using the list-based index
                full_dist = full_dist.reindex(all_cats, fill_value=0)
                train_dist = train_dist.reindex(all_cats, fill_value=0)
                test_dist = test_dist.reindex(all_cats, fill_value=0)
                
                # Add distribution metrics for each category
                for cat in all_cats:
                    add_metric(f"Proportion {col}={cat}",
                             full_dist[cat],
                             train_dist[cat],
                             test_dist[cat])
                    
                    # Add absolute counts
                    cat_count_full = full_counts[cat] if cat in full_counts else 0
                    cat_count_train = train_counts[cat] if cat in train_counts else 0
                    cat_count_test = test_counts[cat] if cat in test_counts else 0
                    
                    add_metric(f"Count {col}={cat}",
                             cat_count_full,
                             cat_count_train,
                             cat_count_test)
            
            # Cramér's V correlations between categorical features
            if len(categorical_columns) > 1:
                full_cramer = cramers_v_matrix(X[categorical_columns], categorical_columns)
                train_cramer = cramers_v_matrix(X_tr[categorical_columns], categorical_columns)
                test_cramer = cramers_v_matrix(X_te[categorical_columns], categorical_columns)
                
                for i, col1 in enumerate(categorical_columns):
                    for j, col2 in enumerate(categorical_columns):
                        if i < j:  # Only show upper triangle
                            add_metric(f"Cramér's V ({col1} vs {col2})",
                                     full_cramer[i, j],
                                     train_cramer[i, j],
                                     test_cramer[i, j])
            
            # Numeric-Categorical associations
            if numerical_columns:
                full_numcat = numeric_categorical_corr_matrix(X[numerical_columns], X[categorical_columns])
                train_numcat = numeric_categorical_corr_matrix(X_tr[numerical_columns], X_tr[categorical_columns])
                test_numcat = numeric_categorical_corr_matrix(X_te[numerical_columns], X_te[categorical_columns])
                
                for i, num_col in enumerate(numerical_columns):
                    for j, cat_col in enumerate(categorical_columns):
                        add_metric(f"Correlation Ratio ({num_col} vs {cat_col})",
                                 full_numcat[i, j],
                                 train_numcat[i, j],
                                 test_numcat[i, j])
        
        # 3. Target distribution with comprehensive statistics
        if np.issubdtype(y.dtype, np.number):
            # Numeric target - comprehensive statistics
            add_metric(f'Target Mean ({target_name})', y.mean(), y_tr.mean(), y_te.mean())
            add_metric(f'Target Std ({target_name})', y.std(), y_tr.std(), y_te.std())
            add_metric(f'Target Skewness ({target_name})', pd.Series(y).skew(), pd.Series(y_tr).skew(), pd.Series(y_te).skew())
            add_metric(f'Target Kurtosis ({target_name})', pd.Series(y).kurtosis(), pd.Series(y_tr).kurtosis(), pd.Series(y_te).kurtosis())
            
            # Percentiles
            percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
            for p in percentiles:
                add_metric(f'Target {p}th Percentile ({target_name})',
                          np.percentile(y, p),
                          np.percentile(y_tr, p),
                          np.percentile(y_te, p))
            
            # Range statistics
            add_metric(f'Target Range ({target_name})', y.max() - y.min(), y_tr.max() - y_tr.min(), y_te.max() - y_te.min())
            add_metric(f'Target IQR ({target_name})', 
                      np.percentile(y, 75) - np.percentile(y, 25),
                      np.percentile(y_tr, 75) - np.percentile(y_tr, 25),
                      np.percentile(y_te, 75) - np.percentile(y_te, 25))
        else:
            # Categorical target with detailed statistics
            y_full_dist = pd.Series(y).value_counts(normalize=True)
            y_tr_dist = pd.Series(y_tr).value_counts(normalize=True)
            y_te_dist = pd.Series(y_te).value_counts(normalize=True)
            
            # Get counts for cardinality metrics
            y_full_counts = pd.Series(y).value_counts()
            y_tr_counts = pd.Series(y_tr).value_counts()
            y_te_counts = pd.Series(y_te).value_counts()
            
            # Add cardinality metrics
            add_metric(f'Target Unique Values Count ({target_name})',
                      len(y_full_counts),
                      len(y_tr_counts),
                      len(y_te_counts))
            
            add_metric(f'Target Most Frequent Value Count ({target_name})',
                      y_full_counts.iloc[0],
                      y_tr_counts.iloc[0] if len(y_tr_counts) > 0 else 0,
                      y_te_counts.iloc[0] if len(y_te_counts) > 0 else 0)
            
            # Convert indices to lists and get unique categories
            all_cats = pd.Index(set(list(y_full_dist.index) + 
                                  list(y_tr_dist.index) + 
                                  list(y_te_dist.index)))
            
            # Reindex using the list-based index
            y_full_dist = y_full_dist.reindex(all_cats, fill_value=0)
            y_tr_dist = y_tr_dist.reindex(all_cats, fill_value=0)
            y_te_dist = y_te_dist.reindex(all_cats, fill_value=0)
            
            # Add distribution metrics for each category
            for cat in all_cats:
                # Add proportions
                add_metric(f'Target Proportion ({target_name}) ({cat})',
                          y_full_dist[cat],
                          y_tr_dist[cat],
                          y_te_dist[cat])
                
                # Add absolute counts
                cat_count_full = y_full_counts[cat] if cat in y_full_counts else 0
                cat_count_train = y_tr_counts[cat] if cat in y_tr_counts else 0
                cat_count_test = y_te_counts[cat] if cat in y_te_counts else 0
                
                add_metric(f'Target Count ({target_name}) ({cat})',
                          cat_count_full,
                          cat_count_train,
                          cat_count_test)
        
        return pd.DataFrame(results).round(4)
        
    except Exception as e:
        logging.error(f"Error in compare_split_distributions: {str(e)}")
        raise


def pareto_indices_vectorized(M):
    """
    Compute indices of non-dominated rows (the Pareto frontier) in matrix M.
    
    Each row in M is a candidate's objective vector. A candidate is considered "better"
    if its objective values are lower (i.e., lower is better). This function uses a 
    vectorized method to compare every candidate against every other candidate.
    
    Parameters:
      M : 2D NumPy array of shape (n, d), where n is the number of candidates and d is
          the number of objective dimensions.
          
    Returns:
      A NumPy array of indices corresponding to the candidates on the Pareto frontier.
    """
    # Expand dimensions so we can compare each candidate to every other.
    M_expanded = M[:, np.newaxis, :]  # Shape: (n, 1, d)
    M_broadcast = M[np.newaxis, :, :]  # Shape: (1, n, d)
    
    # For each pair (i, j), check if candidate j's objectives are <= candidate i's objectives.
    comparison = np.all(M_broadcast <= M_expanded, axis=2)  # Shape: (n, n)
    # Also check if candidate j is strictly better in at least one objective.
    strict = np.any(M_broadcast < M_expanded, axis=2)  # Shape: (n, n)
    # Candidate i is dominated if any other candidate j satisfies both conditions.
    dominated = np.any(comparison & strict, axis=1)
    return np.where(~dominated)[0]


def remove_highly_correlated_features(df, threshold=1):
    """
    Remove numeric columns that are highly correlated based on Spearman correlation.
    
    This function computes the Spearman correlation matrix of the DataFrame 'df' 
    and then examines only the upper triangle (excluding the diagonal) to identify 
    columns that have a correlation greater than the given threshold with any other column.
    
    Parameters:
      df : A pandas DataFrame containing numeric features.
      threshold : A float specifying the correlation threshold; columns with a correlation
                  above this threshold (with any other column) are dropped.
                  
    Returns:
      A tuple (df_reduced, removed_columns) where:
        - df_reduced is the DataFrame with highly correlated columns removed.
        - removed_columns is a list of the names of the columns that were dropped.
    """
    # Compute the absolute Spearman correlation matrix.
    corr_matrix = df.corr(method='spearman').abs()
    # Use only the upper triangle of the correlation matrix.
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Identify columns to drop if any correlation value is above the threshold.
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    if to_drop:
        logging.info(f"Removed highly correlated features: {to_drop}")
    return df.drop(columns=to_drop), to_drop


def compute_robust_cov(df, support_fraction_initial=0.75, max_support_fraction=0.95, step=0.05):
    """
    Compute a robust covariance estimator using the Minimum Covariance Determinant (MinCovDet).
    
    This function dynamically adjusts the support_fraction parameter if warnings occur
    (which indicate instability such as a non full-rank covariance matrix or an increasing determinant).
    
    Parameters:
      df : A pandas DataFrame containing numeric features.
      support_fraction_initial : The initial fraction of data points to use.
      max_support_fraction : The maximum allowed support fraction.
      step : The increment by which to increase support_fraction if warnings occur.
      
    Returns:
      A tuple (mcd, used_support_fraction) where mcd is the fitted MinCovDet instance and 
      used_support_fraction is the final support fraction used.
    """
    support_fraction = support_fraction_initial
    best_mcd = None
    while support_fraction <= max_support_fraction:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                mcd = MinCovDet(support_fraction=support_fraction).fit(df)
            except Exception:
                support_fraction += step
                continue
            # If any warnings are captured, increase the support_fraction.
            if any(issubclass(warn.category, (UserWarning, RuntimeWarning)) for warn in w):
                support_fraction += step
            else:
                best_mcd = mcd
                break
    if best_mcd is None:
        best_mcd = mcd
    return best_mcd, support_fraction


def cramers_v_vec(x, y):
    """
    Compute Cramér's V statistic for two categorical variables (provided as arrays).
    
    Cramér's V is a measure of association between two nominal variables, with values
    between 0 (no association) and 1 (perfect association).
    
    Parameters:
      x : 1D array-like of integer-encoded categorical values.
      y : 1D array-like of integer-encoded categorical values.
      
    Returns:
      A float value representing the strength of association.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    k1 = x.max() + 1
    k2 = y.max() + 1
    # Build the contingency table.
    contingency = np.zeros((k1, k2), dtype=np.int64)
    np.add.at(contingency, (x, y), 1)
    n = contingency.sum()
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    expected = np.outer(row_sums, col_sums) / n
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = ((contingency - expected) ** 2) / expected
        chi2[expected == 0] = 0
    chi2_val = chi2.sum()
    phi2 = chi2_val / n
    r, k = contingency.shape
    if n <= 1:
        return 0.0
    # Apply bias correction.
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denominator = min((kcorr - 1), (rcorr - 1))
    return np.sqrt(phi2corr / denominator) if denominator > 0 else 0.0


def cramers_v_matrix(df, cols):
    """
    Compute a symmetric matrix of Cramér's V statistics for the specified categorical columns.
    Uses vectorized operations for improved performance.
    
    Parameters:
      df : A pandas DataFrame containing categorical features.
      cols : List of column names for which to compute Cramér's V.
      
    Returns:
      A NumPy array of shape (len(cols), len(cols)) containing the Cramér's V values.
    """
    n = len(cols)
    if n == 0:
        return np.array([[]])
        
    # Pre-compute integer codes for all columns at once
    codes_matrix = np.empty((df.shape[0], n), dtype=np.int32)
    max_cats = np.zeros(n, dtype=np.int32)
    
    for i, col in enumerate(cols):
        if hasattr(df[col].dtype, 'name') and df[col].dtype.name == 'category':
            codes = df[col].cat.codes.values
        else:
            codes = pd.factorize(df[col].values)[0]
        codes_matrix[:, i] = codes
        max_cats[i] = codes.max() + 1
    
    # Initialize result matrix
    result = np.eye(n)  # Diagonal is 1.0 by definition
    
    # Compute upper triangle of the matrix
    for i in range(n):
        for j in range(i + 1, n):
            # Get codes for both columns
            codes_i = codes_matrix[:, i]
            codes_j = codes_matrix[:, j]
            
            # Create contingency table using vectorized operations
            k1, k2 = max_cats[i], max_cats[j]
            contingency = np.zeros((k1, k2), dtype=np.int64)
            np.add.at(contingency, (codes_i, codes_j), 1)
            
            # Compute expected frequencies using matrix operations
            n_samples = contingency.sum()
            row_sums = contingency.sum(axis=1)
            col_sums = contingency.sum(axis=0)
            expected = np.outer(row_sums, col_sums) / n_samples
            
            # Compute chi-square statistic using vectorized operations
            with np.errstate(divide='ignore', invalid='ignore'):
                chi2 = np.where(expected > 0,
                              (contingency - expected) ** 2 / expected,
                              0)
            chi2_val = chi2.sum()
            
            # Compute Cramér's V with bias correction
            phi2 = chi2_val / n_samples
            r, k = contingency.shape
            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n_samples - 1))
            rcorr = r - ((r - 1) ** 2) / (n_samples - 1)
            kcorr = k - ((k - 1) ** 2) / (n_samples - 1)
            denominator = min((kcorr - 1), (rcorr - 1))
            
            # Store the result
            if denominator > 0:
                v = np.sqrt(phi2corr / denominator)
            else:
                v = 0.0
            result[i, j] = v
            result[j, i] = v  # Matrix is symmetric
    
    return result


def vectorized_correlation_ratio(fcat, measurements):
    """
    Compute the Pearson correlation ratio (η) between a categorical variable and numeric variables.
    
    Vectorized implementation that can handle multiple numeric variables simultaneously.
    
    Parameters:
      fcat : 1D array of integer-encoded categorical values
      measurements : 2D array of shape (n_samples, n_features) for multiple numeric variables
                    or 1D array for a single numeric variable
      
    Returns:
      Array of correlation ratios, one per numeric variable
    """
    measurements = np.asarray(measurements)
    if measurements.ndim == 1:
        measurements = measurements.reshape(-1, 1)
    
    # Compute overall means and total sum of squares
    overall_means = measurements.mean(axis=0)
    ss_total = np.sum((measurements - overall_means) ** 2, axis=0)
    
    # Handle zero variance
    zero_var_mask = ss_total == 0
    if np.all(zero_var_mask):
        return np.zeros(measurements.shape[1])
    
    # Compute group statistics using matrix operations
    unique_cats, inverse_indices = np.unique(fcat, return_inverse=True)
    n_groups = len(unique_cats)
    
    # Create sparse group indicator matrix
    group_matrix = np.zeros((len(fcat), n_groups))
    group_matrix[np.arange(len(fcat)), inverse_indices] = 1
    
    # Compute group counts and sums efficiently
    group_counts = group_matrix.sum(axis=0)
    group_sums = group_matrix.T @ measurements
    
    # Compute group means and between sum of squares
    with np.errstate(divide='ignore', invalid='ignore'):
        group_means = group_sums / group_counts[:, np.newaxis]
        group_means = np.nan_to_num(group_means, 0)
    
    # Compute between sum of squares using broadcasting
    ss_between = np.sum(
        group_counts[:, np.newaxis] * (group_means - overall_means) ** 2,
        axis=0
    )
    
    # Calculate correlation ratio with handling for zero variance
    result = np.zeros_like(ss_total, dtype=float)
    nonzero_mask = ~zero_var_mask
    result[nonzero_mask] = np.sqrt(ss_between[nonzero_mask] / ss_total[nonzero_mask])
    
    return result.squeeze()


def numeric_categorical_corr_matrix(numerical_df, categorical_df):
    """
    Compute a matrix of correlation ratios between numeric and categorical features.
    
    This function efficiently computes correlation ratios between all pairs of numeric
    and categorical features using vectorized operations. It handles categorical encoding,
    correlation computation, and matrix operations in a memory-efficient way.
    
    Parameters:
      numerical_df : DataFrame containing numeric features
      categorical_df : DataFrame containing categorical features
      
    Returns:
      NumPy array of shape (number of numeric features, number of categorical features)
      where each element (i,j) represents the correlation ratio between the i-th numeric
      feature and the j-th categorical feature.
    
    Notes:
      - Uses vectorized operations for better performance
      - Handles both category and non-category dtypes
      - Pre-allocates arrays for memory efficiency
      - Processes all pairs simultaneously for speed
    """
    num_cols = numerical_df.columns
    cat_cols = categorical_df.columns
    n_num = len(num_cols)
    n_cat = len(cat_cols)
    
    # Convert all numeric columns to a single array for efficiency
    numeric_data = numerical_df.values
    
    # Precompute all categorical encodings at once for better performance
    cat_codes = np.empty((categorical_df.shape[0], n_cat), dtype=np.int32)
    for j, col in enumerate(cat_cols):
        if hasattr(categorical_df[col].dtype, 'name') and categorical_df[col].dtype.name == 'category':
            cat_codes[:, j] = categorical_df[col].cat.codes.values
        else:
            cat_codes[:, j] = pd.factorize(categorical_df[col].values)[0]
    
    # Initialize result matrix with proper dimensions
    result = np.empty((n_num, n_cat))
    
    # Process each categorical variable efficiently
    for j in range(n_cat):
        # Compute correlation ratios for all numeric variables at once
        result[:, j] = vectorized_correlation_ratio(cat_codes[:, j], numeric_data)
    
    return result




# %%
###############################################################################
# Main Function: find_best_split_seed
###############################################################################
def find_best_split_seed(
    X, 
    y, 
    categorical_columns=None, 
    numerical_columns=None, 
    n_samples=100, 
    test_size=0.2, 
    weights=None,  # Not used in this multi-objective version.
    n_cv_splits=1, 
    objective_functions=None,
    verbose=False, 
    save_results_file=None,
    random_search_seed=None
):
    """
    Finds the best random seed for splitting the dataset into training and testing sets,
    aiming to preserve overall distributional and correlation properties.
    
    Six objectives are computed:
      1. Robust_Mahalanobis_Numeric:
         Robust Mahalanobis distance (using MinCovDet) between the overall numeric center and
         the means of the training and test sets.
      2. Wasserstein_Numeric:
         Average Wasserstein distance (per numeric column) between the overall and split distributions.
      3. JSD_Categorical:
         Average Jensen–Shannon divergence between the overall and split categorical distributions.
      4. Spearman_Correlation_Diff:
         Frobenius norm difference between the overall and split Spearman correlation matrices for numeric data.
      5. CramersV_Correlation_Diff:
         Frobenius norm difference between the overall and split categorical association matrices (via Cramér’s V).
      6. NumCat_Association_Diff:
         Frobenius norm difference between the overall and split numeric–categorical association matrices.
    
    After evaluation, the raw objective vectors are robustly normalized (using the median and MAD).
    The Pareto frontier is computed using a vectorized method, and candidates are sorted in ascending
    order by their sum of normalized scores (sum_z). The candidate with the lowest sum_z is selected as the best.
    
    Reference Ranges:
      For each objective, the range is given as a list [lower_bound, upper_bound] with values formatted to 4 decimals.
      The "percentage_of_range" indicates the relative position of the best candidate's value within this range,
      where 0% means near the optimum (lowest) observed value.
    
    Returns:
      A dictionary with:
        - 'best_seed': The selected seed, with its sum_z in parentheses.
        - 'best_objectives': A dictionary mapping descriptive measure names to the best candidate's raw values.
        - 'reference_info': A dictionary with range and percentage info for each objective.
        - 'pareto_frontier': A list (ordered ascending by sum_z) of candidate seeds, each with its sum_z and raw objective vector.
        - 'duration_seconds': Total execution time.
        - 'raw_objectives': The raw objective matrix from all candidate seeds.
    """
    # Use default weights if not provided.
    if weights is None:
        weights = {'numerical': 1.0, 'spearman': 1.0, 'cramers': 1.0, 'numcat': 1.0}
    
    # Determine columns if not provided.
    if categorical_columns is None and numerical_columns is None:
        raise ValueError("At least one of 'categorical_columns' or 'numerical_columns' must be provided.")
    if categorical_columns is None:
        categorical_columns = [col for col in X.columns if col not in numerical_columns]
    if numerical_columns is None:
        numerical_columns = [col for col in X.columns if col not in categorical_columns]
    
    # Fill missing values in numeric columns.
    for col in numerical_columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    # Fill missing values in categorical columns.
    for col in categorical_columns:
        if X[col].isnull().any():
            mode_val = X[col].mode()[0]
            X[col] = X[col].fillna(mode_val)
        if X[col].dtype.name != 'category':
            X[col] = X[col].astype('category')
    
    # Ensure y is one-dimensional.
    if isinstance(y, np.ndarray):
        if y.ndim > 1:
            y = y.flatten()
    elif isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.squeeze()
    
    # Check if target is numeric.
    is_target_numeric = np.issubdtype(y.dtype, np.number)
    is_target_categorical = not is_target_numeric
    
    if is_target_categorical:
        overall_target_dist = y.value_counts(normalize=True).sort_index()


    ############################################################################
    # (1) Remove Highly Correlated Numeric Features using Spearman correlation.
    ############################################################################
    if numerical_columns:
        X_numeric = X[numerical_columns]
        X_numeric_reduced, removed_cols = remove_highly_correlated_features(X_numeric)
        numerical_columns = X_numeric_reduced.columns.tolist()
        X = X.copy()
        for col in removed_cols:
            if col in X.columns:
                X.drop(columns=col, inplace=True)
    
    ############################################################################
    # (2) Dynamic Robust Covariance Estimation.
    ############################################################################
    if numerical_columns:
        if is_target_numeric:
            overall_num_df = pd.concat([X[numerical_columns], pd.Series(y, name='target')], axis=1)
        else:
            overall_num_df = X[numerical_columns]
        mcd, chosen_support = compute_robust_cov(overall_num_df, support_fraction_initial=0.75)
        robust_mean = mcd.location_
        robust_cov = mcd.covariance_
        inv_cov = np.linalg.pinv(robust_cov)
        overall_num_dists = {col: X[col].values for col in numerical_columns}
    else:
        robust_mean = None
        inv_cov = None
        overall_num_dists = {}

    if categorical_columns:
        overall_cat_dists = {col: X[col].value_counts(normalize=True).sort_index() for col in categorical_columns}
    else:
        overall_cat_dists = {}

    if numerical_columns:
        if overall_num_df.shape[1] > 1:
            overall_corr_num = spearmanr(overall_num_df).correlation
        else:
            overall_corr_num = np.array([[1.0]])
    else:
        overall_corr_num = None

    if categorical_columns:
        if is_target_categorical:
            overall_cat_df = pd.concat([X[categorical_columns], pd.Series(y, name='target')], axis=1)
            cat_cols = categorical_columns + ['target']
        else:
            overall_cat_df = X[categorical_columns]
            cat_cols = categorical_columns
        overall_corr_cat = cramers_v_matrix(overall_cat_df, cat_cols)
    else:
        overall_corr_cat = None

    if numerical_columns and categorical_columns:
        overall_numcat = numeric_categorical_corr_matrix(X[numerical_columns], X[categorical_columns])
    else:
        overall_numcat = None

    ############################################################################
    # (3) Define Default Objective Functions.
    ############################################################################
    if objective_functions is None:
        def obj_robust_mahalanobis(X_tr, X_te, y_tr, y_te):
            """
            Compute robust Mahalanobis distances between train/test means and overall mean.
            
            This function efficiently computes the Mahalanobis distances using matrix operations
            and the pre-computed robust covariance matrix. It handles both numeric and categorical
            target variables appropriately.
            
            Parameters:
                X_tr: Training set features
                X_te: Test set features
                y_tr: Training set target
                y_te: Test set target
                
            Returns:
                float: Sum of Mahalanobis distances for train and test sets
                
            Notes:
                - Uses pre-computed robust mean and inverse covariance matrix
                - Handles numerical stability with non-negative distance enforcement
                - Efficiently combines target variable when numeric
            """
            if is_target_numeric:
                train_df = pd.concat([X_tr[numerical_columns], pd.Series(y_tr, name='target')], axis=1)
                test_df = pd.concat([X_te[numerical_columns], pd.Series(y_te, name='target')], axis=1)
            else:
                train_df = X_tr[numerical_columns]
                test_df = X_te[numerical_columns]
            
            # Compute means efficiently
            train_mean = train_df.mean(axis=0).values
            test_mean = test_df.mean(axis=0).values
            
            # Compute Mahalanobis distances with numerical stability
            d_train = np.sqrt(max(0, (train_mean - robust_mean).T.dot(inv_cov).dot(train_mean - robust_mean)))
            d_test = np.sqrt(max(0, (test_mean - robust_mean).T.dot(inv_cov).dot(test_mean - robust_mean)))
            
            return d_train + d_test

        def obj_wasserstein(X_tr, X_te, y_tr, y_te):
            """
            Vectorized Wasserstein distance computation for all numeric columns.
            Uses matrix operations to compute distances efficiently.
            """
            # Convert all data to numpy arrays at once
            train_data = X_tr[numerical_columns].values
            test_data = X_te[numerical_columns].values
            overall_data = np.vstack([overall_num_dists[col] for col in numerical_columns]).T
            
            # Pre-allocate array for distances
            distances = np.zeros(len(numerical_columns))
            
            # Compute distances using vectorized operations
            for i in range(len(numerical_columns)):
                # Sort the arrays once for each column
                train_sorted = np.sort(train_data[:, i])
                test_sorted = np.sort(test_data[:, i])
                overall_sorted = np.sort(overall_data[:, i])
                
                # Compute CDFs using vectorized operations
                train_cdf = np.linspace(0, 1, len(train_sorted))
                test_cdf = np.linspace(0, 1, len(test_sorted))
                overall_cdf = np.linspace(0, 1, len(overall_sorted))
                
                # Interpolate to common grid for comparison
                grid = np.unique(np.concatenate([train_sorted, test_sorted, overall_sorted]))
                train_interp = np.interp(grid, train_sorted, train_cdf)
                test_interp = np.interp(grid, test_sorted, test_cdf)
                overall_interp = np.interp(grid, overall_sorted, overall_cdf)
                
                # Compute Wasserstein distances using trapezoid rule
                distances[i] = (
                    np.trapezoid(np.abs(train_interp - overall_interp), grid) +
                    np.trapezoid(np.abs(test_interp - overall_interp), grid)
                )
            
            return np.mean(distances)

        # Compute JSD using vectorized operations
        # JSD(P||Q) = 0.5 * (KL(P||M) + KL(Q||M)) where M = 0.5 * (P + Q)
        def vectorized_jsd(p, q):
            # Compute mixture distribution
            m = 0.5 * (p + q)
            # Handle zero probabilities
            m = np.where(m == 0, np.finfo(float).eps, m)
            p = np.where(p == 0, np.finfo(float).eps, p)
            q = np.where(q == 0, np.finfo(float).eps, q)
            # Compute KL divergences
            kl_pm = np.sum(p * np.log(p / m))
            kl_qm = np.sum(q * np.log(q / m))
            return np.sqrt(0.5 * (kl_pm + kl_qm))
        
        def obj_jsd(X_tr, X_te, y_tr, y_te):
            """
            Vectorized Jensen-Shannon divergence computation for all categorical columns.
            Uses matrix operations for efficient probability computation.
            """
            if not categorical_columns:
                return 0
                
            # Pre-allocate array for distances
            distances = np.zeros(len(categorical_columns))
            
            # Process each categorical column using vectorized operations
            for i, col in enumerate(categorical_columns):
                # Get unique categories and their counts for all three distributions at once
                all_cats = overall_cat_dists[col].index
                n_cats = len(all_cats)
                
                # Create a mapping from categories to indices
                cat_to_idx = {cat: idx for idx, cat in enumerate(all_cats)}
                
                # Initialize probability arrays
                probs = np.zeros((3, n_cats))  # [overall, train, test]
                
                # Vectorized computation of probability distributions
                def get_prob_vector(series):
                    # Count occurrences of each category
                    counts = np.zeros(n_cats)
                    unique, counts_arr = np.unique(series, return_counts=True)
                    for cat, count in zip(unique, counts_arr):
                        if cat in cat_to_idx:
                            counts[cat_to_idx[cat]] = count
                    # Normalize to get probabilities
                    return counts / len(series)
                
                # Fill probability arrays efficiently
                probs[0] = np.array([overall_cat_dists[col].get(cat, 0) for cat in all_cats])
                probs[1] = get_prob_vector(X_tr[col])
                probs[2] = get_prob_vector(X_te[col])
                
                                
                # Compute total JSD for this column
                distances[i] = (
                    vectorized_jsd(probs[0], probs[1]) +  # JSD between overall and train
                    vectorized_jsd(probs[0], probs[2])    # JSD between overall and test
                )
            
            return np.mean(distances)

        
        
        
        def obj_spearman_corr(X_tr, X_te, y_tr, y_te):
            """
            Vectorized Spearman correlation computation using matrix operations.
            
            Parameters:
                X_tr: Training set features
                X_te: Test set features
                y_tr: Training set target
                y_te: Test set target
                
            Returns:
                float: Sum of Frobenius norms between train/test correlation matrices and overall correlation matrix
                
            Notes:
                - Uses vectorized operations for rank computation
                - Handles numerical stability with proper error handling
                - Efficiently computes correlations using matrix operations
            """
            try:
                if not numerical_columns:
                    return 0
                    
                # Prepare data
                if is_target_numeric:
                    train_data = np.column_stack([X_tr[numerical_columns].values, y_tr])
                    test_data = np.column_stack([X_te[numerical_columns].values, y_te])
                else:
                    train_data = X_tr[numerical_columns].values
                    test_data = X_te[numerical_columns].values
                    
                if train_data.shape[1] <= 1:
                    return 0
                    
                # Vectorized rank computation
                def fast_rankdata(data):
                    n_samples, n_features = data.shape
                    ranks = np.zeros_like(data)
                    
                    # Compute ranks for each feature using matrix operations
                    for j in range(n_features):
                        col = data[:, j]
                        # Get sorting indices
                        order = np.argsort(col)
                        # Create rank array
                        rank = np.zeros(n_samples)
                        # Handle ties by averaging
                        pos = 0
                        while pos < n_samples:
                            start = pos
                            val = col[order[pos]]
                            while pos < n_samples and col[order[pos]] == val:
                                pos += 1
                            rank[order[start:pos]] = 0.5 * (start + pos - 1)
                        ranks[:, j] = rank
                    
                    return ranks
                    
                # Compute ranks
                train_ranks = fast_rankdata(train_data)
                test_ranks = fast_rankdata(test_data)
                
                # Compute correlations using matrix operations
                def fast_corr(ranks):
                    n_samples = ranks.shape[0]
                    # Center the ranks
                    ranks_centered = ranks - np.mean(ranks, axis=0)
                    # Compute correlation matrix using matrix multiplication
                    corr = np.dot(ranks_centered.T, ranks_centered) / (n_samples - 1)
                    # Normalize
                    std = np.std(ranks, axis=0, ddof=1)
                    corr /= np.outer(std, std)
                    return corr
                    
                train_corr = fast_corr(train_ranks)
                test_corr = fast_corr(test_ranks)
                
                # Compute Frobenius norms
                return norm(train_corr - overall_corr_num, ord='fro') + norm(test_corr - overall_corr_num, ord='fro')
            except Exception as e:
                logging.error(f"Error in obj_spearman_corr: {str(e)}")
                return np.inf

        def obj_cramersv_corr(X_tr, X_te, y_tr, y_te):
            """
            Compute Cramér's V correlation differences between train/test and overall data.
            
            Parameters:
                X_tr: Training set features
                X_te: Test set features
                y_tr: Training set target
                y_te: Test set target
                
            Returns:
                float: Sum of Frobenius norms between train/test Cramér's V matrices and overall Cramér's V matrix
                
            Notes:
                - Handles both categorical and non-categorical target variables
                - Uses efficient matrix operations for correlation computation
                - Returns 0 if no categorical columns are present
            """
            try:
                if categorical_columns:
                    if is_target_categorical:
                        train_cat = pd.concat([X_tr[categorical_columns], pd.Series(y_tr, name='target')], axis=1)
                        test_cat = pd.concat([X_te[categorical_columns], pd.Series(y_te, name='target')], axis=1)
                        current_cols = categorical_columns + ['target']
                    else:
                        train_cat = X_tr[categorical_columns]
                        test_cat = X_te[categorical_columns]
                        current_cols = categorical_columns
                    train_corr_cat = cramers_v_matrix(train_cat, current_cols)
                    test_corr_cat = cramers_v_matrix(test_cat, current_cols)
                    return norm(train_corr_cat - overall_corr_cat, ord='fro') + norm(test_corr_cat - overall_corr_cat, ord='fro')
                else:
                    return 0
            except Exception as e:
                logging.error(f"Error in obj_cramersv_corr: {str(e)}")
                return np.inf

        def obj_numcat_assoc(X_tr, X_te, y_tr, y_te):
            """
            Compute numeric-categorical association differences between train/test and overall data.
            
            Parameters:
                X_tr: Training set features
                X_te: Test set features
                y_tr: Training set target (unused)
                y_te: Test set target (unused)
                
            Returns:
                float: Sum of Frobenius norms between train/test association matrices and overall association matrix
                
            Notes:
                - Computes correlation ratios between numeric and categorical features
                - Uses efficient matrix operations for computation
                - Returns 0 if either numerical or categorical columns are missing
            """
            try:
                if numerical_columns and categorical_columns:
                    train_assoc = numeric_categorical_corr_matrix(X_tr[numerical_columns], X_tr[categorical_columns])
                    test_assoc = numeric_categorical_corr_matrix(X_te[numerical_columns], X_te[categorical_columns])
                    return norm(train_assoc - overall_numcat, ord='fro') + norm(test_assoc - overall_numcat, ord='fro')
                else:
                    return 0
            except Exception as e:
                logging.error(f"Error in obj_numcat_assoc: {str(e)}")
                return np.inf

        objective_functions = {
            "Robust_Mahalanobis_Numeric": obj_robust_mahalanobis,
            "Wasserstein_Numeric": obj_wasserstein,
            "JSD_Categorical": obj_jsd,
            "Spearman_Correlation_Diff": obj_spearman_corr,
            "CramersV_Correlation_Diff": obj_cramersv_corr,
            "NumCat_Association_Diff": obj_numcat_assoc
        }

    ############################################################################
    # (4) Compute Objective Vector for a Given Split.
    ############################################################################
    def compute_split_objectives(X_tr, X_te, y_tr, y_te, obj_funcs):
        """
        Compute the objective vector for a given train/test split.
        
        Iterates over all objective functions and returns:
          - A NumPy array of objective values (ordered by sorted objective names).
          - A dictionary mapping each objective name to its computed value.
          - The list of sorted objective names.
        """
        objs = {}
        for name, func in obj_funcs.items():
            objs[name] = func(X_tr, X_te, y_tr, y_te)
        keys = sorted(obj_funcs.keys())
        obj_vector = np.array([objs[k] for k in keys])
        return obj_vector, objs, keys

    ############################################################################
    # (5) Evaluate Candidate Seeds and Compute Pareto Frontier.
    ############################################################################
    def evaluate_seed(seed_val):
        """
        Evaluate a candidate seed by performing ShuffleSplit cross-validation and
        computing the average objective vector over the splits.
        
        Parameters:
          seed_val: The candidate seed value.
          
        Returns:
          A tuple (seed_val, avg_obj_vector, breakdown) where breakdown is empty.
        """
        try:
            cv = ShuffleSplit(n_splits=n_cv_splits, test_size=test_size, random_state=seed_val)
            obj_list = []
            for train_idx, test_idx in cv.split(X):
                X_tr = X.iloc[train_idx]
                X_te = X.iloc[test_idx]
                y_tr = (y.iloc[train_idx] if isinstance(y, (pd.Series, pd.DataFrame)) else y[train_idx]).squeeze()
                y_te = (y.iloc[test_idx] if isinstance(y, (pd.Series, pd.DataFrame)) else y[test_idx]).squeeze()
                obj, _, _ = compute_split_objectives(X_tr, X_te, y_tr, y_te, objective_functions)
                obj_list.append(obj)
            avg_obj = np.mean(obj_list, axis=0)
            return seed_val, avg_obj, {}
        except Exception as e:
            logging.error(f"Error evaluating seed {seed_val}: {e}")
            return seed_val, np.full(len(objective_functions), np.inf), {}

    def random_search(seeds):
        """
        Evaluate all candidate seeds in parallel using ThreadPoolExecutor with optimized
        batch processing and error handling.
        
        Parameters:
          seeds: A list of candidate seed values.
          
        Returns:
          A list of tuples (seed, objective_vector, breakdown).
        """
        # Determine optimal batch size and number of workers
        n_workers = min(32, (os.cpu_count() or 1) * 2)
        batch_size = max(1, len(seeds) // (n_workers * 4))  # Ensure enough batches per worker
        
        # Pre-allocate result lists with thread-safe containers
        candidates = []
        failed_seeds = []
        
        def process_batch(batch_seeds):
            """Process a batch of seeds with proper error handling."""
            batch_results = []
            batch_failures = []
            for seed in batch_seeds:
                try:
                    result = evaluate_seed(seed)
                    if not np.all(np.isfinite(result[1])):  # Check for invalid objectives
                        raise ValueError(f"Invalid objective values for seed {seed}")
                    batch_results.append(result)
                except Exception as e:
                    logging.warning(f"Failed to evaluate seed {seed}: {str(e)}")
                    batch_failures.append(seed)
            return batch_results, batch_failures
        
        try:
            # Split seeds into batches
            seed_batches = [seeds[i:i + batch_size] for i in range(0, len(seeds), batch_size)]
            
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(process_batch, batch): batch 
                    for batch in seed_batches
                }
                
                # Process results as they complete
                if USE_TQDM:
                    futures = tqdm(
                        concurrent.futures.as_completed(future_to_batch),
                        total=len(seed_batches),
                        desc="Random Search"
                    )
                else:
                    futures = concurrent.futures.as_completed(future_to_batch)
                
                for future in futures:
                    try:
                        batch_results, batch_failures = future.result(timeout=300)  # 5 minute timeout per batch
                        candidates.extend(batch_results)
                        failed_seeds.extend(batch_failures)
                    except concurrent.futures.TimeoutError:
                        batch = future_to_batch[future]
                        logging.error(f"Batch {batch} timed out")
                        failed_seeds.extend(batch)
                    except Exception as e:
                        batch = future_to_batch[future]
                        logging.error(f"Error processing batch {batch}: {str(e)}")
                        failed_seeds.extend(batch)
        
        except Exception as e:
            logging.error(f"Error in parallel processing: {str(e)}")
            raise
        
        # Log statistics
        total_seeds = len(seeds)
        success_rate = (len(candidates) / total_seeds) * 100
        failure_rate = (len(failed_seeds) / total_seeds) * 100
        
        logging.info(f"Evaluation completed:")
        logging.info(f"  - Total seeds: {total_seeds}")
        logging.info(f"  - Successful: {len(candidates)} ({success_rate:.1f}%)")
        logging.info(f"  - Failed: {len(failed_seeds)} ({failure_rate:.1f}%)")
        
        if not candidates:
            raise RuntimeError("No valid candidates were found during random search")
        
        return candidates

    # Start the timer for performance measurement.
    logging.info("Starting Random Search...")
    start_time = time.perf_counter()
    if random_search_seed is not None:
        random.seed(random_search_seed)
    random_seeds = [random.randint(0, 10_000) for _ in range(n_samples)]
    candidates = random_search(random_seeds)
    duration = time.perf_counter() - start_time
    logging.info(f"Execution Time: {duration:.2f} seconds")
    
    # Collect raw objective vectors from all candidate seeds.
    obj_matrix = np.array([cand[1] for cand in candidates])
    # Robust normalization: use median and MAD.
    medians = np.median(obj_matrix, axis=0)
    mads = np.median(np.abs(obj_matrix - medians), axis=0)
    mads = np.where(mads == 0, 1, mads)
    obj_z = (obj_matrix - medians) / mads
    
    # Compute Pareto frontier indices.
    pareto_idx = pareto_indices_vectorized(obj_z)
    sum_z_array = np.sum(obj_z[pareto_idx], axis=1)
    # Order Pareto candidates in ascending order of sum_z.
    order = np.argsort(sum_z_array)
    # Extract Pareto candidates.
    pareto_idx_sorted = pareto_idx[order]
    pareto_candidates_sorted = [candidates[i] for i in pareto_idx_sorted]
    sum_z_sorted = np.sum(obj_z[pareto_idx_sorted], axis=1)
    
    # Select the best candidate: one with the minimal sum_z.
    best_idx_in_pareto = np.argmin(sum_z_sorted)
    best_seed, best_obj, _ = pareto_candidates_sorted[best_idx_in_pareto]
    
    logging.info(f"Best seed: {best_seed}")
    
    # Mapping internal objective keys to descriptive labels.
    descriptive_labels = {
        "Robust_Mahalanobis_Numeric": "Robust Mahalanobis Distance (Numeric)",
        "Wasserstein_Numeric": "Avg Wasserstein Distance (Numeric)",
        "JSD_Categorical": "Avg Jensen–Shannon Divergence (Categorical)",
        "Spearman_Correlation_Diff": "Spearman Correlation Diff (Numeric)",
        "CramersV_Correlation_Diff": "Cramér’s V Diff (Categorical)",
        "NumCat_Association_Diff": "Numeric-Categorical Assoc Diff"
    }
    
    # Sorted objective keys (alphabetically) as used in our objective vector.
    ref_measures = sorted(objective_functions.keys())
    # Build the best objective dictionary with descriptive labels.
    best_obj_dict = {descriptive_labels[k]: float(best_obj[i]) for i, k in enumerate(ref_measures)}
    
    # Compute reference bounds and percentage positions for each objective.
    reference_info = {}
    for i, measure in enumerate(ref_measures):
        lower = np.min(obj_matrix[:, i])
        upper = np.max(obj_matrix[:, i])
        range_val = upper - lower if (upper - lower) != 0 else 1
        best_pct = 100 * (best_obj[i] - lower) / range_val
        # Format the range as a list with 4 decimal places.
        reference_info[descriptive_labels[measure]] = {
            "range": [float(f"{lower:.4f}"), float(f"{upper:.4f}")],
            "best_value": float(f"{best_obj[i]:.4f}"),
            "percentage_of_range": float(f"{best_pct:.1f}")
        }
    
    # Build Pareto frontier output with seed, sum_z, and raw objective vector (formatted to 4 decimals).
    pareto_output = []
    for idx in pareto_idx_sorted:
        seed_val = candidates[idx][0]
        s_z = float(np.sum(obj_z[idx]))
        # Format the raw objective vector.
        raw_obj = list(np.around(candidates[idx][1], 4))
        pareto_output.append({"seed": seed_val, "sum_z": s_z, "objective_raw": [float(x) for x in raw_obj]})
         
    result = {
        'best_seed': best_seed,
        'best_objectives': best_obj_dict,
        'reference_info': reference_info,
        'pareto_frontier': pareto_output,
        'duration_seconds': duration,
        'raw_objectives': obj_matrix.tolist()
    }
    if verbose:
        result['verbose_details'] = {
            'median_objectives': medians.tolist(),
            'mad_objectives': mads.tolist(),
            'all_candidates': [{
                'seed': cand[0],
                'objective': cand[1].tolist()
            } for cand in candidates]
        }
        
        # Retrieve best candidate information from the Pareto frontier.
        best_candidate = [cand for cand in result['pareto_frontier'] if cand['seed'] == result['best_seed']][0]

        print(f"\nBest Objective Vector (raw) with reference measures and bounds (sum_z = {best_candidate['sum_z']:.4f}):")
        for measure, info in result['reference_info'].items():
            print(f"  {measure}: {info['best_value']:.4f} (Range: {info['range']}, {info['percentage_of_range']:.1f}% of range)")

        print("\nPareto Frontier Candidates (ordered ascending by sum_z):")
        for cand in result['pareto_frontier']:
            print(f"  Seed {cand['seed']} with sum_z = {cand['sum_z']:.4f} and raw objective vector = {cand['objective_raw']}")

    if save_results_file:
        try:
            with open(save_results_file, 'w') as f:
                json.dump(result, f, indent=4)
            logging.info(f"Results saved to {save_results_file}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")
    
    return result





# %%
###############################################################################
# Example Usage
###############################################################################

# Load data from an Excel file and perform initial cleaning.
data = pd.read_csv(r'C:\Data\Fish Market\Fish.csv')
data_cleaned = data.clean_names()

data_cleaned

# %%
# Separate features (X) and target (y).
target_col_name = 'weight'
X = data_cleaned.drop(target_col_name, axis=1)
y = data_cleaned[[target_col_name]]

# Define categorical and numerical columns.
categorical_columns = [ 'species' ]

# Call the main function to find the best seed.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f'best_split_seed_results_{timestamp}.json'

res = find_best_split_seed(
    X=X,
    y=y,
    categorical_columns=categorical_columns,
    numerical_columns=None,
    n_samples=200,
    test_size=0.2,
    weights={'numerical': 1.0, 'spearman': 1.0, 'cramers': 1.0, 'numcat': 1.0},
    n_cv_splits=5,
    verbose=True,
    save_results_file=results_file,
    random_search_seed=4245
)

# %%
compare_df = compare_split_distributions(X, y, res['best_seed'], test_size=0.2, 
                                categorical_columns=categorical_columns, numerical_columns=None)
compare_df

# %%
