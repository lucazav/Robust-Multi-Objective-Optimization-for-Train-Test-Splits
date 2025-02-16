import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
import logging
from scipy.stats import spearmanr

def remove_highly_correlated_features(df, threshold=1):
    """
    Remove numeric columns that are highly correlated based on Spearman correlation.
    """
    corr_matrix = df.corr(method='spearman').abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    if to_drop:
        logging.info(f"Removed highly correlated features: {to_drop}")
    return df.drop(columns=to_drop), to_drop

def compute_robust_cov(df, support_fraction_initial=0.75, max_support_fraction=0.95, step=0.05):
    """
    Compute a robust covariance estimator using the Minimum Covariance Determinant (MinCovDet).
    """
    import warnings
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
            if any(issubclass(warn.category, (UserWarning, RuntimeWarning)) for warn in w):
                support_fraction += step
            else:
                best_mcd = mcd
                break
    if best_mcd is None:
        best_mcd = mcd
    return best_mcd, support_fraction

def pareto_indices_vectorized(M):
    """
    Compute indices of non-dominated rows (the Pareto frontier) in matrix M.
    """
    M_expanded = M[:, np.newaxis, :]
    M_broadcast = M[np.newaxis, :, :]
    comparison = np.all(M_broadcast <= M_expanded, axis=2)
    strict = np.any(M_broadcast < M_expanded, axis=2)
    dominated = np.any(comparison & strict, axis=1)
    return np.where(~dominated)[0]

def cramers_v_vec(x, y):
    """
    Compute Cramér's V statistic for two categorical variables.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    k1 = x.max() + 1
    k2 = y.max() + 1
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
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denominator = min((kcorr - 1), (rcorr - 1))
    return np.sqrt(phi2corr / denominator) if denominator > 0 else 0.0

def cramers_v_matrix(df, cols):
    """
    Compute a symmetric matrix of Cramér's V statistics for categorical columns.
    """
    n = len(cols)
    if n == 0:
        return np.array([[]])
        
    codes_matrix = np.empty((df.shape[0], n), dtype=np.int32)
    max_cats = np.zeros(n, dtype=np.int32)
    
    for i, col in enumerate(cols):
        if hasattr(df[col].dtype, 'name') and df[col].dtype.name == 'category':
            codes = df[col].cat.codes.values
        else:
            codes = pd.factorize(df[col].values)[0]
        codes_matrix[:, i] = codes
        max_cats[i] = codes.max() + 1
    
    result = np.eye(n)
    
    for i in range(n):
        for j in range(i + 1, n):
            codes_i = codes_matrix[:, i]
            codes_j = codes_matrix[:, j]
            
            k1, k2 = max_cats[i], max_cats[j]
            contingency = np.zeros((k1, k2), dtype=np.int64)
            np.add.at(contingency, (codes_i, codes_j), 1)
            
            n_samples = contingency.sum()
            row_sums = contingency.sum(axis=1)
            col_sums = contingency.sum(axis=0)
            expected = np.outer(row_sums, col_sums) / n_samples
            
            with np.errstate(divide='ignore', invalid='ignore'):
                chi2 = np.where(expected > 0,
                              (contingency - expected) ** 2 / expected,
                              0)
            chi2_val = chi2.sum()
            
            phi2 = chi2_val / n_samples
            r, k = contingency.shape
            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n_samples - 1))
            rcorr = r - ((r - 1) ** 2) / (n_samples - 1)
            kcorr = k - ((k - 1) ** 2) / (n_samples - 1)
            denominator = min((kcorr - 1), (rcorr - 1))
            
            if denominator > 0:
                v = np.sqrt(phi2corr / denominator)
            else:
                v = 0.0
            result[i, j] = v
            result[j, i] = v
    
    return result

def vectorized_correlation_ratio(fcat, measurements):
    """
    Compute the Pearson correlation ratio between categorical and numeric variables.
    """
    measurements = np.asarray(measurements)
    if measurements.ndim == 1:
        measurements = measurements.reshape(-1, 1)
    
    overall_means = measurements.mean(axis=0)
    ss_total = np.sum((measurements - overall_means) ** 2, axis=0)
    
    zero_var_mask = ss_total == 0
    if np.all(zero_var_mask):
        return np.zeros(measurements.shape[1])
    
    unique_cats, inverse_indices = np.unique(fcat, return_inverse=True)
    n_groups = len(unique_cats)
    
    group_matrix = np.zeros((len(fcat), n_groups))
    group_matrix[np.arange(len(fcat)), inverse_indices] = 1
    
    group_counts = group_matrix.sum(axis=0)
    group_sums = group_matrix.T @ measurements
    
    with np.errstate(divide='ignore', invalid='ignore'):
        group_means = group_sums / group_counts[:, np.newaxis]
        group_means = np.nan_to_num(group_means, 0)
    
    ss_between = np.sum(
        group_counts[:, np.newaxis] * (group_means - overall_means) ** 2,
        axis=0
    )
    
    result = np.zeros_like(ss_total, dtype=float)
    nonzero_mask = ~zero_var_mask
    result[nonzero_mask] = np.sqrt(ss_between[nonzero_mask] / ss_total[nonzero_mask])
    
    return result.squeeze()

def numeric_categorical_corr_matrix(numerical_df, categorical_df):
    """
    Compute correlation matrix between numeric and categorical features.
    """
    num_cols = numerical_df.columns
    cat_cols = categorical_df.columns
    n_num = len(num_cols)
    n_cat = len(cat_cols)
    
    numeric_data = numerical_df.values
    
    cat_codes = np.empty((categorical_df.shape[0], n_cat), dtype=np.int32)
    for j, col in enumerate(cat_cols):
        if hasattr(categorical_df[col].dtype, 'name') and categorical_df[col].dtype.name == 'category':
            cat_codes[:, j] = categorical_df[col].cat.codes.values
        else:
            cat_codes[:, j] = pd.factorize(categorical_df[col].values)[0]
    
    result = np.empty((n_num, n_cat))
    
    for j in range(n_cat):
        result[:, j] = vectorized_correlation_ratio(cat_codes[:, j], numeric_data)
    
    return result
