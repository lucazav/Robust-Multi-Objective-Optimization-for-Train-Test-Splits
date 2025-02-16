import os
import numpy as np
import pandas as pd
import random
import time
import logging
import json
import warnings
import inspect
import concurrent.futures
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import spearmanr
from numpy.linalg import norm

from .utils import (
    remove_highly_correlated_features,
    compute_robust_cov,
    pareto_indices_vectorized,
    cramers_v_matrix,
    numeric_categorical_corr_matrix
)

from sklearn.model_selection import ShuffleSplit

def compare_split_distributions(X, y, best_seed, test_size=0.2, 
                               categorical_columns=None, numerical_columns=None):
    """
    Compare key distribution metrics between the full dataset, training set, and test set
    for the best found seed. Returns a DataFrame with metrics for easy comparison.
    
    Parameters:
    -----------
    X : pandas DataFrame
        The feature matrix
    y : array-like
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
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if not isinstance(test_size, float) or not 0 < test_size < 1:
            raise ValueError("test_size must be a float between 0 and 1")
        if isinstance(y, pd.DataFrame):
            target_name = y.columns[0]
            y = y.squeeze()
            
        if categorical_columns is None and numerical_columns is None:
            numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_columns = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
        elif categorical_columns is None:
            categorical_columns = [col for col in X.columns if col not in numerical_columns]
        elif numerical_columns is None:
            numerical_columns = [col for col in X.columns if col not in categorical_columns]
            
        splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=best_seed)
        train_idx, test_idx = next(splitter.split(X))
        
        X_tr = X.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_tr = y.iloc[train_idx] if isinstance(y, (pd.Series, pd.DataFrame)) else y[train_idx]
        y_te = y.iloc[test_idx] if isinstance(y, (pd.Series, pd.DataFrame)) else y[test_idx]
        
        results = {
            'Metric': [],
            'Full Dataset': [],
            'Training Set': [],
            'Test Set': []
        }
        
        def add_metric(name, full_val, train_val, test_val):
            results['Metric'].append(name)
            results['Full Dataset'].append(full_val)
            results['Training Set'].append(train_val)
            results['Test Set'].append(test_val)
        
        if numerical_columns:
            for col in numerical_columns:
                add_metric(f"Mean ({col})", 
                          X[col].mean(),
                          X_tr[col].mean(),
                          X_te[col].mean())
                
                add_metric(f"Std ({col})",
                          X[col].std(),
                          X_tr[col].std(),
                          X_te[col].std())
                
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
                
                full_sorted = np.sort(X[col].values)
                train_sorted = np.sort(X_tr[col].values)
                test_sorted = np.sort(X_te[col].values)
                
                full_cdf = np.linspace(0, 1, len(full_sorted))
                train_cdf = np.linspace(0, 1, len(train_sorted))
                test_cdf = np.linspace(0, 1, len(test_sorted))
                
                grid = np.unique(np.concatenate([full_sorted, train_sorted, test_sorted]))
                full_interp = np.interp(grid, full_sorted, full_cdf)
                train_interp = np.interp(grid, train_sorted, train_cdf)
                test_interp = np.interp(grid, test_sorted, test_cdf)
                
                train_wasserstein = np.trapezoid(np.abs(train_interp - full_interp), grid)
                test_wasserstein = np.trapezoid(np.abs(test_interp - full_interp), grid)
                
                add_metric(f"Wasserstein Distance ({col})",
                          0.0,
                          train_wasserstein,
                          test_wasserstein)
            
            if len(numerical_columns) > 1:
                full_corr = spearmanr(X[numerical_columns]).correlation
                train_corr = spearmanr(X_tr[numerical_columns]).correlation
                test_corr = spearmanr(X_te[numerical_columns]).correlation
                
                for i, col1 in enumerate(numerical_columns):
                    for j, col2 in enumerate(numerical_columns):
                        if i < j:
                            add_metric(f"Spearman Correlation ({col1} vs {col2})",
                                     full_corr[i, j],
                                     train_corr[i, j],
                                     test_corr[i, j])
        
        if categorical_columns:
            for col in categorical_columns:
                full_dist = pd.Series(X[col]).value_counts(normalize=True)
                train_dist = pd.Series(X_tr[col]).value_counts(normalize=True)
                test_dist = pd.Series(X_te[col]).value_counts(normalize=True)
                
                full_counts = pd.Series(X[col]).value_counts()
                train_counts = pd.Series(X_tr[col]).value_counts()
                test_counts = pd.Series(X_te[col]).value_counts()
                
                add_metric(f"Unique Values Count ({col})",
                          len(full_counts),
                          len(train_counts),
                          len(test_counts))
                
                add_metric(f"Most Frequent Value Count ({col})",
                          full_counts.iloc[0],
                          train_counts.iloc[0] if len(train_counts) > 0 else 0,
                          test_counts.iloc[0] if len(test_counts) > 0 else 0)
                
                all_cats = pd.Index(set(list(full_dist.index) + 
                                      list(train_dist.index) + 
                                      list(test_dist.index)))
                
                full_dist = full_dist.reindex(all_cats, fill_value=0)
                train_dist = train_dist.reindex(all_cats, fill_value=0)
                test_dist = test_dist.reindex(all_cats, fill_value=0)
                
                for cat in all_cats:
                    add_metric(f"Proportion {col}={cat}",
                             full_dist[cat],
                             train_dist[cat],
                             test_dist[cat])
                    
                    cat_count_full = full_counts[cat] if cat in full_counts else 0
                    cat_count_train = train_counts[cat] if cat in train_counts else 0
                    cat_count_test = test_counts[cat] if cat in test_counts else 0
                    
                    add_metric(f"Count {col}={cat}",
                             cat_count_full,
                             cat_count_train,
                             cat_count_test)
            
            if len(categorical_columns) > 1:
                full_cramer = cramers_v_matrix(X[categorical_columns], categorical_columns)
                train_cramer = cramers_v_matrix(X_tr[categorical_columns], categorical_columns)
                test_cramer = cramers_v_matrix(X_te[categorical_columns], categorical_columns)
                
                for i, col1 in enumerate(categorical_columns):
                    for j, col2 in enumerate(categorical_columns):
                        if i < j:
                            add_metric(f"Cramér's V ({col1} vs {col2})",
                                     full_cramer[i, j],
                                     train_cramer[i, j],
                                     test_cramer[i, j])
            
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
        
        if np.issubdtype(y.dtype, np.number):
            add_metric(f'Target Mean ({target_name})', y.mean(), y_tr.mean(), y_te.mean())
            add_metric(f'Target Std ({target_name})', y.std(), y_tr.std(), y_te.std())
            add_metric(f'Target Skewness ({target_name})', pd.Series(y).skew(), pd.Series(y_tr).skew(), pd.Series(y_te).skew())
            add_metric(f'Target Kurtosis ({target_name})', pd.Series(y).kurtosis(), pd.Series(y_tr).kurtosis(), pd.Series(y_te).kurtosis())
            
            percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
            for p in percentiles:
                add_metric(f'Target {p}th Percentile ({target_name})',
                          np.percentile(y, p),
                          np.percentile(y_tr, p),
                          np.percentile(y_te, p))
            
            add_metric(f'Target Range ({target_name})', y.max() - y.min(), y_tr.max() - y_tr.min(), y_te.max() - y_te.min())
            add_metric(f'Target IQR ({target_name})', 
                      np.percentile(y, 75) - np.percentile(y, 25),
                      np.percentile(y_tr, 75) - np.percentile(y_tr, 25),
                      np.percentile(y_te, 75) - np.percentile(y_te, 25))
        else:
            y_full_dist = pd.Series(y).value_counts(normalize=True)
            y_tr_dist = pd.Series(y_tr).value_counts(normalize=True)
            y_te_dist = pd.Series(y_te).value_counts(normalize=True)
            
            y_full_counts = pd.Series(y).value_counts()
            y_tr_counts = pd.Series(y_tr).value_counts()
            y_te_counts = pd.Series(y_te).value_counts()
            
            add_metric(f'Target Unique Values Count ({target_name})',
                      len(y_full_counts),
                      len(y_tr_counts),
                      len(y_te_counts))
            
            add_metric(f'Target Most Frequent Value Count ({target_name})',
                      y_full_counts.iloc[0],
                      y_tr_counts.iloc[0] if len(y_tr_counts) > 0 else 0,
                      y_te_counts.iloc[0] if len(y_te_counts) > 0 else 0)
            
            all_cats = pd.Index(set(list(y_full_dist.index) + 
                                  list(y_tr_dist.index) + 
                                  list(y_te_dist.index)))
            
            y_full_dist = y_full_dist.reindex(all_cats, fill_value=0)
            y_tr_dist = y_tr_dist.reindex(all_cats, fill_value=0)
            y_te_dist = y_te_dist.reindex(all_cats, fill_value=0)
            
            for cat in all_cats:
                add_metric(f'Target Proportion ({target_name}) ({cat})',
                          y_full_dist[cat],
                          y_tr_dist[cat],
                          y_te_dist[cat])
                
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

# Optional: tqdm for progress bars (if available)
try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_best_split_seed(
    X, 
    y,
    splitter,
    categorical_columns=None, 
    numerical_columns=None, 
    n_samples=100, 
    weights=None,
    n_cv_splits=1, 
    objective_functions=None,
    verbose=False, 
    save_results_file=None,
    random_search_seed=None,
    groups=None
):
    """
    Finds the best random seed for splitting the dataset into training and testing sets,
    aiming to preserve overall distributional and correlation properties.
    
    Parameters:
    -----------
    X : pandas DataFrame
        The feature matrix
    y : array-like
        The target variable
    splitter : sklearn.model_selection._split.BaseCrossValidator
        A scikit-learn CV splitter object (e.g., ShuffleSplit, StratifiedShuffleSplit,
        GroupShuffleSplit). The splitter must be initialized with test_size and
        random_state parameters.
    categorical_columns : list of str, optional
        Names of categorical columns
    numerical_columns : list of str, optional
        Names of numerical columns
    n_samples : int, default=100
        Number of random seeds to evaluate
    weights : dict, optional
        Not used in this multi-objective version
    n_cv_splits : int, default=1
        Number of cross-validation splits to average over
    objective_functions : dict, optional
        Custom objective functions to use
    verbose : bool, default=False
        Whether to print detailed information
    save_results_file : str, optional
        Path to save results as JSON
    random_search_seed : int, optional
        Random seed for reproducibility
    groups : array-like, optional
        Group labels for the samples. Required when using GroupShuffleSplit or
        StratifiedGroupKFold splitters.
        
    Returns:
    --------
    dict
        A dictionary containing the best seed and related metrics
    """
    # Validate splitter has required attributes
    if not hasattr(splitter, 'random_state'):
        raise ValueError("Splitter must have a random_state parameter")
    if not hasattr(splitter, 'split'):
        raise ValueError("Splitter must have a split method")
        
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

    # Remove highly correlated numeric features
    if numerical_columns:
        X_numeric = X[numerical_columns]
        X_numeric_reduced, removed_cols = remove_highly_correlated_features(X_numeric)
        numerical_columns = X_numeric_reduced.columns.tolist()
        X = X.copy()
        for col in removed_cols:
            if col in X.columns:
                X.drop(columns=col, inplace=True)
    
    # Dynamic robust covariance estimation
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

    # Define default objective functions
    if objective_functions is None:
        def obj_robust_mahalanobis(X_tr, X_te, y_tr, y_te):
            if is_target_numeric:
                train_df = pd.concat([X_tr[numerical_columns], pd.Series(y_tr, name='target')], axis=1)
                test_df = pd.concat([X_te[numerical_columns], pd.Series(y_te, name='target')], axis=1)
            else:
                train_df = X_tr[numerical_columns]
                test_df = X_te[numerical_columns]
            
            train_mean = train_df.mean(axis=0).values
            test_mean = test_df.mean(axis=0).values
            
            d_train = np.sqrt(max(0, (train_mean - robust_mean).T.dot(inv_cov).dot(train_mean - robust_mean)))
            d_test = np.sqrt(max(0, (test_mean - robust_mean).T.dot(inv_cov).dot(test_mean - robust_mean)))
            
            return d_train + d_test

        def obj_wasserstein(X_tr, X_te, y_tr, y_te):
            train_data = X_tr[numerical_columns].values
            test_data = X_te[numerical_columns].values
            overall_data = np.vstack([overall_num_dists[col] for col in numerical_columns]).T
            
            distances = np.zeros(len(numerical_columns))
            
            for i in range(len(numerical_columns)):
                train_sorted = np.sort(train_data[:, i])
                test_sorted = np.sort(test_data[:, i])
                overall_sorted = np.sort(overall_data[:, i])
                
                train_cdf = np.linspace(0, 1, len(train_sorted))
                test_cdf = np.linspace(0, 1, len(test_sorted))
                overall_cdf = np.linspace(0, 1, len(overall_sorted))
                
                grid = np.unique(np.concatenate([train_sorted, test_sorted, overall_sorted]))
                train_interp = np.interp(grid, train_sorted, train_cdf)
                test_interp = np.interp(grid, test_sorted, test_cdf)
                overall_interp = np.interp(grid, overall_sorted, overall_cdf)
                
                distances[i] = (
                    np.trapezoid(np.abs(train_interp - overall_interp), grid) +
                    np.trapezoid(np.abs(test_interp - overall_interp), grid)
                )
            
            return np.mean(distances)

        def vectorized_jsd(p, q):
            m = 0.5 * (p + q)
            m = np.where(m == 0, np.finfo(float).eps, m)
            p = np.where(p == 0, np.finfo(float).eps, p)
            q = np.where(q == 0, np.finfo(float).eps, q)
            kl_pm = np.sum(p * np.log(p / m))
            kl_qm = np.sum(q * np.log(q / m))
            return np.sqrt(0.5 * (kl_pm + kl_qm))
        
        def obj_jsd(X_tr, X_te, y_tr, y_te):
            if not categorical_columns:
                return 0
                
            distances = np.zeros(len(categorical_columns))
            
            for i, col in enumerate(categorical_columns):
                all_cats = overall_cat_dists[col].index
                n_cats = len(all_cats)
                cat_to_idx = {cat: idx for idx, cat in enumerate(all_cats)}
                probs = np.zeros((3, n_cats))
                
                def get_prob_vector(series):
                    counts = np.zeros(n_cats)
                    unique, counts_arr = np.unique(series, return_counts=True)
                    for cat, count in zip(unique, counts_arr):
                        if cat in cat_to_idx:
                            counts[cat_to_idx[cat]] = count
                    return counts / len(series)
                
                probs[0] = np.array([overall_cat_dists[col].get(cat, 0) for cat in all_cats])
                probs[1] = get_prob_vector(X_tr[col])
                probs[2] = get_prob_vector(X_te[col])
                
                distances[i] = (
                    vectorized_jsd(probs[0], probs[1]) +
                    vectorized_jsd(probs[0], probs[2])
                )
            
            return np.mean(distances)

        def obj_spearman_corr(X_tr, X_te, y_tr, y_te):
            try:
                if not numerical_columns:
                    return 0
                    
                if is_target_numeric:
                    train_data = np.column_stack([X_tr[numerical_columns].values, y_tr])
                    test_data = np.column_stack([X_te[numerical_columns].values, y_te])
                else:
                    train_data = X_tr[numerical_columns].values
                    test_data = X_te[numerical_columns].values
                    
                if train_data.shape[1] <= 1:
                    return 0
                    
                def fast_rankdata(data):
                    n_samples, n_features = data.shape
                    ranks = np.zeros_like(data)
                    
                    for j in range(n_features):
                        col = data[:, j]
                        order = np.argsort(col)
                        rank = np.zeros(n_samples)
                        pos = 0
                        while pos < n_samples:
                            start = pos
                            val = col[order[pos]]
                            while pos < n_samples and col[order[pos]] == val:
                                pos += 1
                            rank[order[start:pos]] = 0.5 * (start + pos - 1)
                        ranks[:, j] = rank
                    
                    return ranks
                    
                train_ranks = fast_rankdata(train_data)
                test_ranks = fast_rankdata(test_data)
                
                def fast_corr(ranks):
                    n_samples = ranks.shape[0]
                    ranks_centered = ranks - np.mean(ranks, axis=0)
                    corr = np.dot(ranks_centered.T, ranks_centered) / (n_samples - 1)
                    std = np.std(ranks, axis=0, ddof=1)
                    corr /= np.outer(std, std)
                    return corr
                    
                train_corr = fast_corr(train_ranks)
                test_corr = fast_corr(test_ranks)
                
                return norm(train_corr - overall_corr_num, ord='fro') + norm(test_corr - overall_corr_num, ord='fro')
            except Exception as e:
                logging.error(f"Error in obj_spearman_corr: {str(e)}")
                return np.inf

        def obj_cramersv_corr(X_tr, X_te, y_tr, y_te):
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

    def compute_split_objectives(X_tr, X_te, y_tr, y_te, obj_funcs):
        objs = {}
        for name, func in obj_funcs.items():
            objs[name] = func(X_tr, X_te, y_tr, y_te)
        keys = sorted(obj_funcs.keys())
        obj_vector = np.array([objs[k] for k in keys])
        return obj_vector, objs, keys

    def evaluate_seed(seed_val):
        try:
            # Set the random state of the splitter
            splitter.random_state = seed_val
            
            obj_list = []
            # Handle different splitter types
            split_args = [X]
            
            # Inspect the signature of splitter's split method
            sig = inspect.signature(splitter.split)

            # Convert the parameters to a dict for easy checking
            split_params = sig.parameters.keys()
            
            if 'y' in split_params and y is not None:
                split_args.append(y)
            if 'groups' in split_params and groups is not None:
                split_args.append(groups)
                
            for train_idx, test_idx in splitter.split(*split_args):
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
        n_workers = min(32, (os.cpu_count() or 1) * 2)
        batch_size = max(1, len(seeds) // (n_workers * 4))
        
        candidates = []
        failed_seeds = []
        
        def process_batch(batch_seeds):
            batch_results = []
            batch_failures = []
            for seed in batch_seeds:
                try:
                    result = evaluate_seed(seed)
                    if not np.all(np.isfinite(result[1])):
                        raise ValueError(f"Invalid objective values for seed {seed}")
                    batch_results.append(result)
                except Exception as e:
                    logging.warning(f"Failed to evaluate seed {seed}: {str(e)}")
                    batch_failures.append(seed)
            return batch_results, batch_failures
        
        try:
            seed_batches = [seeds[i:i + batch_size] for i in range(0, len(seeds), batch_size)]
            
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                future_to_batch = {
                    executor.submit(process_batch, batch): batch 
                    for batch in seed_batches
                }
                
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
                        batch_results, batch_failures = future.result(timeout=300)
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
        "CramersV_Correlation_Diff": "Cramér's V Diff (Categorical)",
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
