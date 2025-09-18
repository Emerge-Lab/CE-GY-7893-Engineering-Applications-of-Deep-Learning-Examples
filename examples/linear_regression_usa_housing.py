# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Linear Regression: Three Implementation Approaches
#
# This notebook demonstrates linear regression using three different approaches:
# 1. **Scikit-learn**
# 2. **Closed-form solution** - The analytical approach (Normal Equation)
# 3. **Gradient descent** - The optimization approach (generally wouldn't do this one, it's just to build gradient descent intuition)
#
# We'll use the USA Housing dataset to predict house prices based on various features.

# %% [markdown]
# ## Import Required Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import kagglehub
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# %% [markdown]
# ## Load USA Housing Dataset
#
# We'll use the USA Housing dataset from Kaggle which contains housing data
# with features like average area income, house age, number of rooms, etc.

# %%
# Download the USA Housing dataset from Kaggle
print("Downloading USA Housing dataset from Kaggle...")
path = kagglehub.dataset_download("vedavyasv/usa-housing")
print("Path to dataset files:", path)

# %%
# Load the dataset
import os
dataset_files = os.listdir(path)
print("Files in dataset:", dataset_files)

# Load the CSV file
csv_files = [f for f in dataset_files if f.endswith('.csv')]
if csv_files:
    dataset_file = csv_files[0]  # Take the first CSV file
    df = pd.read_csv(os.path.join(path, dataset_file))
    print(f"\nLoaded dataset: {dataset_file}")
    print(f"Dataset shape: {df.shape}")
    print(f"Column names: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
else:
    print("No CSV files found in the dataset directory")

# %%
# Examine the dataset structure
print("Dataset Info:")
print(df.info())
print(f"\nDataset shape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic statistics:")
print(df.describe())

# %% [markdown]
# ## Data Preprocessing and Exploration

# %%
# Clean column names (remove spaces, make lowercase)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print("Cleaned column names:", list(df.columns))

# Display sample of the data
print(f"\nSample of cleaned data:")
print(df.head())

# %%
# Identify target and features
# The target is typically 'price' in housing datasets
target_candidates = ['price', 'house_price', 'home_price', 'value']
target_col = None

for candidate in target_candidates:
    if candidate in df.columns:
        target_col = candidate
        break

# If no obvious target found, use the last column (common in housing datasets)
if target_col is None:
    target_col = df.columns[-1]

print(f"Using '{target_col}' as target variable")

# Get all numeric features except target
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numeric_features:
    numeric_features.remove(target_col)

print(f"Numeric features: {numeric_features}")
print(f"Target variable: {target_col}")

# Remove any non-numeric or problematic columns
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical features: {categorical_features}")

# %%
# Handle missing values and prepare data
df_clean = df.copy()

# Remove rows with missing target values
df_clean = df_clean.dropna(subset=[target_col])

# Handle missing values in features (fill with median)
for col in numeric_features:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

print(f"Clean dataset shape: {df_clean.shape}")
print(f"Missing values after cleaning:\n{df_clean[numeric_features + [target_col]].isnull().sum()}")

# %% [markdown]
# ## Exploratory Data Analysis

# %%
# Target variable analysis
print(f"Target variable ({target_col}) statistics:")
print(df_clean[target_col].describe())

# Plot target distribution
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(df_clean[target_col], bins=50, alpha=0.7, edgecolor='black')
plt.title(f'Distribution of {target_col}')
plt.xlabel(target_col)
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.boxplot(df_clean[target_col])
plt.title(f'Boxplot of {target_col}')
plt.ylabel(target_col)

plt.subplot(1, 3, 3)
# Log scale if values are very skewed
if df_clean[target_col].min() > 0:
    plt.hist(np.log1p(df_clean[target_col]), bins=50, alpha=0.7, edgecolor='black')
    plt.title(f'Log Distribution of {target_col}')
    plt.xlabel(f'log({target_col})')
else:
    plt.scatter(range(len(df_clean)), df_clean[target_col], alpha=0.5)
    plt.title(f'{target_col} vs Index')
    plt.xlabel('Index')

plt.tight_layout()
plt.show()

# %%
# Feature analysis and correlation
print(f"Feature statistics:")
print(df_clean[numeric_features].describe())

# Correlation analysis
correlation_data = df_clean[numeric_features + [target_col]]

# Correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = correlation_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, fmt='.3f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Show correlations with target
target_correlations = correlation_matrix[target_col].abs().sort_values(ascending=False)
print(f"\nFeatures most correlated with {target_col}:")
print(target_correlations[1:])  # Exclude self-correlation

# %%
# Scatter plots of features vs target
n_features = len(numeric_features)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
if n_rows == 1:
    axes = axes.reshape(1, -1)

for i, feature in enumerate(numeric_features):
    row = i // n_cols
    col = i % n_cols
    
    axes[row, col].scatter(df_clean[feature], df_clean[target_col], alpha=0.5)
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel(target_col)
    axes[row, col].set_title(f'{feature} vs {target_col}')
    
    # Add trend line
    z = np.polyfit(df_clean[feature], df_clean[target_col], 1)
    p = np.poly1d(z)
    axes[row, col].plot(df_clean[feature], p(df_clean[feature]), "r--", alpha=0.8)

# Remove empty subplots
for i in range(n_features, n_rows * n_cols):
    row = i // n_cols
    col = i % n_cols
    fig.delaxes(axes[row, col])

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Prepare Data for Modeling

# %%
# Prepare features and target
X = df_clean[numeric_features].values
y = df_clean[target_col].values

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Feature names: {numeric_features}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# %% [markdown]
# ### Data Standardization (Z-Score Normalization)
# If you look at the data exploration, you'll notice that some of the values are huge: in the millions! If we take the square of that, our loss will be gigantic. It'll put us into issues of numerical instability (running out of numbers on the computer). So, very often, we'll do what's called "standardization" of the features.
#
# **Why standardization is crucial for linear regression:**
# 1. **Numerical stability**: Prevents ill-conditioned matrices
# 2. **Fair feature comparison**: Features with different scales won't dominate
# 3. **Gradient descent convergence**: Ensures similar learning rates for all features

# %% [markdown]
# ### Explicit Standardization Implementation
#
# Let's implement standardization from scratch to understand exactly what it means:
#
# **Standardization Formula (Z-Score):**
# $$z = \frac{x - \mu}{\sigma}$$
#
# Where:
# - $x$ = original feature value
# - $\mu$ = mean of the feature
# - $\sigma$ = standard deviation of the feature
# - $z$ = standardized feature value

# %%
# Explicit standardization implementation
def standardize_features(X_train, X_test=None):
    """
    Explicitly standardize features by subtracting mean and dividing by std
    
    Returns:
        X_train_standardized, X_test_standardized, feature_means, feature_stds
    """
    # Step 1: Calculate mean and std from training data
    feature_means = np.mean(X_train, axis=0)
    feature_stds = np.std(X_train, axis=0, ddof=1)  # Use sample std (ddof=1)
    
    # Step 2: Apply standardization transformation
    # z = (x - μ) / σ
    X_train_standardized = (X_train - feature_means) / feature_stds
    
    if X_test is not None:
        # Use training statistics for test set (important!)
        X_test_standardized = (X_test - feature_means) / feature_stds
        return X_train_standardized, X_test_standardized, feature_means, feature_stds
    
    return X_train_standardized, feature_means, feature_stds

def standardize_target(y_train, y_test=None):
    """
    Explicitly standardize target variable
    """
    target_mean = np.mean(y_train)
    target_std = np.std(y_train, ddof=1)
    
    y_train_standardized = (y_train - target_mean) / target_std
    
    if y_test is not None:
        y_test_standardized = (y_test - target_mean) / target_std
        return y_train_standardized, y_test_standardized, target_mean, target_std
    
    return y_train_standardized, target_mean, target_std

def inverse_standardize_target(y_standardized, target_mean, target_std):
    """
    Transform standardized predictions back to original scale
    """
    return y_standardized * target_std + target_mean

# Apply explicit standardization
print("Applying Explicit Standardization...")
X_train_scaled, X_test_scaled, feature_means, feature_stds = standardize_features(X_train, X_test)
y_train_scaled, y_test_scaled, target_mean, target_std = standardize_target(y_train, y_test)

print(f"Feature Standardization Results:")
print(f"=" * 40)
print(f"Original feature means: {X_train.mean(axis=0).round(2)}")
print(f"Original feature stds:  {X_train.std(axis=0, ddof=1).round(2)}")
print(f"Computed feature means: {feature_means.round(2)}")
print(f"Computed feature stds:  {feature_stds.round(2)}")

print(f"\nAfter standardization:")
print(f"Standardized feature means: {X_train_scaled.mean(axis=0).round(6)}")  # Should be ~0
print(f"Standardized feature stds:  {X_train_scaled.std(axis=0, ddof=1).round(6)}")   # Should be ~1

print(f"\nTarget Variable Standardization:")
print(f"Original target mean: {y_train.mean():.2f}, std: {y_train.std(ddof=1):.2f}")
print(f"Computed target mean: {target_mean:.2f}, std: {target_std:.2f}")
print(f"Standardized target mean: {y_train_scaled.mean():.6f}, std: {y_train_scaled.std(ddof=1):.6f}")

# %% [markdown]
# ### Step-by-Step Standardization Demonstration
#
# Let's see standardization in action for a single feature:

# %%
# Demonstrate standardization step-by-step for first feature
feature_idx = 0
feature_name = numeric_features[feature_idx]

print(f"Standardization demonstration for feature: '{feature_name}'")
print("=" * 60)

# Original values (first 10 samples)
original_values = X_train[:10, feature_idx]
print(f"Original values (first 10): {original_values.round(2)}")

# Step 1: Calculate statistics
mean_val = feature_means[feature_idx]
std_val = feature_stds[feature_idx]
print(f"\nStep 1 - Calculate statistics:")
print(f"  Mean (μ): {mean_val:.2f}")
print(f"  Std (σ):  {std_val:.2f}")

# Step 2: Apply standardization formula
print(f"\nStep 2 - Apply standardization: z = (x - μ) / σ")
standardized_values = X_train_scaled[:10, feature_idx]
for i in range(5):  # Show first 5 calculations
    original = original_values[i]
    standardized = standardized_values[i]
    manual_calc = (original - mean_val) / std_val
    print(f"  Sample {i+1}: ({original:.2f} - {mean_val:.2f}) / {std_val:.2f} = {manual_calc:.3f} ≈ {standardized:.3f}")

print(f"\nStandardized values (first 10): {standardized_values.round(3)}")
print(f"Verification - Standardized mean: {standardized_values.mean():.6f} (should be ~0)")
print(f"Verification - Standardized std:  {standardized_values.std(ddof=1):.6f} (should be ~1)")

# %% [markdown]
# ## Method 1: Scikit-learn Linear Regression

# %%
# Train scikit-learn linear regression on standardized data
print("Training Scikit-learn Linear Regression on Standardized Data...")
sklearn_lr = LinearRegression()
sklearn_lr.fit(X_train_scaled, y_train_scaled)

# Make predictions (in scaled space)
sklearn_pred_train_scaled = sklearn_lr.predict(X_train_scaled)
sklearn_pred_test_scaled = sklearn_lr.predict(X_test_scaled)

# Transform predictions back to original scale using our explicit function
sklearn_pred_train = inverse_standardize_target(sklearn_pred_train_scaled, target_mean, target_std)
sklearn_pred_test = inverse_standardize_target(sklearn_pred_test_scaled, target_mean, target_std)

# Calculate metrics in original scale
sklearn_train_mse = mean_squared_error(y_train, sklearn_pred_train)
sklearn_test_mse = mean_squared_error(y_test, sklearn_pred_test)
sklearn_train_r2 = r2_score(y_train, sklearn_pred_train)
sklearn_test_r2 = r2_score(y_test, sklearn_pred_test)

# Also calculate metrics in scaled space for comparison
sklearn_train_mse_scaled = mean_squared_error(y_train_scaled, sklearn_pred_train_scaled)
sklearn_test_mse_scaled = mean_squared_error(y_test_scaled, sklearn_pred_test_scaled)

print(f"Scikit-learn Results (Original Scale):")
print(f"  Train MSE: {sklearn_train_mse:.2f}")
print(f"  Test MSE:  {sklearn_test_mse:.2f}")
print(f"  Train R²:  {sklearn_train_r2:.4f}")
print(f"  Test R²:   {sklearn_test_r2:.4f}")

print(f"\nScikit-learn Results (Scaled Space):")
print(f"  Train MSE: {sklearn_train_mse_scaled:.6f}")
print(f"  Test MSE:  {sklearn_test_mse_scaled:.6f}")
print(f"  Coefficients (scaled): {sklearn_lr.coef_}")
print(f"  Intercept (scaled): {sklearn_lr.intercept_:.6f}")  # Should be ~0

# %% [markdown]
# ## Method 2: Closed-Form Solution (Normal Equation)
#
# The closed-form solution for linear regression is:
# $$\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$
#
# This gives us the optimal weights directly without iteration.

# %%
class LinearRegressionClosedForm:
    """Linear Regression using the Normal Equation (Closed-Form Solution)"""
    
    def __init__(self, method='solve'):
        self.weights = None
        self.intercept = None
        self.method = method  # 'solve', 'pinv', or 'inv'
        
    def fit(self, X, y):
        """Fit the model using the normal equation"""
        # Add bias term (intercept) by adding a column of ones
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        if self.method == 'solve':
            # Method 1: Linear solver (RECOMMENDED)
            # Solve: (X^T X) w = X^T y
            # This is more numerically stable and efficient than computing the inverse
            XTX = X_with_bias.T @ X_with_bias
            XTy = X_with_bias.T @ y
            self.weights_with_bias = np.linalg.solve(XTX, XTy)
            
        elif self.method == 'pinv':
            # Method 2: Pseudo-inverse (handles singular matrices i.e. noninvertible matrices correctly but is less efficient)
            self.weights_with_bias = np.linalg.pinv(X_with_bias) @ y
            
        elif self.method == 'inv':
            # Method 3: Direct matrix inversion (NOT RECOMMENDED - numerically unstable)
            try:
                XTX = X_with_bias.T @ X_with_bias
                XTX_inv = np.linalg.inv(XTX)
                self.weights_with_bias = XTX_inv @ X_with_bias.T @ y
            except np.linalg.LinAlgError:
                print("Matrix is singular, falling back to pseudo-inverse")
                self.weights_with_bias = np.linalg.pinv(X_with_bias) @ y
        
        # Separate intercept and weights
        self.intercept = self.weights_with_bias[0]
        self.weights = self.weights_with_bias[1:]
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        return X @ self.weights + self.intercept

# %% [markdown]
# ### Comparing Different Closed-Form Approaches
#
# Let's compare three ways to solve the normal equation and understand their trade-offs:

# %%
import time

# Compare different closed-form methods
methods_closed = ['solve', 'pinv', 'inv']
closed_form_results = {}

print("Comparing Closed-Form Solution Methods:")
print("=" * 50)

for method in methods_closed:
    print(f"\nTesting method: {method}")
    
    # Time the training
    start_time = time.time()
    model = LinearRegressionClosedForm(method=method)
    model.fit(X_train_scaled, y_train_scaled)  # Use standardized data
    training_time = time.time() - start_time
    
    # Make predictions (in scaled space then transform back)
    pred_test_scaled = model.predict(X_test_scaled)
    pred_test = inverse_standardize_target(pred_test_scaled, target_mean, target_std)
    
    # Calculate metrics in original scale
    mse = mean_squared_error(y_test, pred_test)
    r2 = r2_score(y_test, pred_test)
    
    closed_form_results[method] = {
        'model': model,
        'mse': mse,
        'r2': r2,
        'time': training_time,
        'weights': model.weights.copy(),
        'intercept': model.intercept,
        'pred_test': pred_test,
        'pred_test_scaled': pred_test_scaled
    }
    
    print(f"  Training time: {training_time:.6f} seconds")
    print(f"  Test MSE: {mse:.2f}")
    print(f"  Test R²: {r2:.4f}")

# %% [markdown]
# ### Understanding the Differences
#
# 1. **Linear Solver (`np.linalg.solve`)** - RECOMMENDED
#    - Solves the system $(X^T X) w = X^T y$ directly
#    - Most numerically stable and efficient
#    - Uses LU decomposition internally
#    - Fails only if matrix is truly singular
#
# 2. **Pseudo-inverse (`np.linalg.pinv`)**
#    - Computes $w = X^{\dagger} y$ where $X^{\dagger}$ is the pseudo-inverse
#    - Handles singular/rank-deficient matrices
#    - More expensive computationally (uses SVD)
#    - Always produces a solution (minimum norm solution for underdetermined systems)
#
# 3. **Direct Inversion (`np.linalg.inv`)** - NOT RECOMMENDED
#    - Explicitly computes $(X^T X)^{-1}$ then multiplies
#    - Numerically unstable (amplifies rounding errors)
#    - More expensive than solve
#    - Fails for singular matrices

# %%
# Use the linear solver results for the rest of the notebook
print(f"\nUsing Linear Solver method for remaining analysis...")
closedform_lr = closed_form_results['solve']['model']

# Get the already computed predictions (properly scaled)
closedform_pred_test = closed_form_results['solve']['pred_test']
closedform_test_mse = closed_form_results['solve']['mse']
closedform_test_r2 = closed_form_results['solve']['r2']

# Make train predictions with proper scaling
closedform_pred_train_scaled = closedform_lr.predict(X_train_scaled)
closedform_pred_train = inverse_standardize_target(closedform_pred_train_scaled, target_mean, target_std)

# Calculate train metrics
closedform_train_mse = mean_squared_error(y_train, closedform_pred_train)
closedform_train_r2 = r2_score(y_train, closedform_pred_train)

print(f"Final Closed-Form Results (Linear Solver):")
print(f"  Train MSE: {closedform_train_mse:.2f}")
print(f"  Test MSE:  {closedform_test_mse:.2f}")
print(f"  Train R²:  {closedform_train_r2:.4f}")
print(f"  Test R²:   {closedform_test_r2:.4f}")

# %% [markdown]
# ### Benefits of Standardization
#
# By standardizing our data, we've achieved:
# 1. **Better numerical stability**: Lower condition numbers
# 2. **Faster convergence**: Gradient descent converges more reliably
# 3. **Fair feature weighting**: All features contribute equally to the optimization
# 4. **Consistent scaling**: Coefficients are now comparable across features
# 5. **Near-zero intercept**: In scaled space, intercept should be ~0 (since target is centered)

# %% [markdown]
# ## Method 3: Gradient Descent Implementation
#
# Gradient descent iteratively updates weights using:
# $$\mathbf{w} = \mathbf{w} - \alpha \frac{\partial J}{\partial \mathbf{w}}$$
#
# Where the gradient for linear regression is:
# $$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m} \mathbf{X}^T (\mathbf{X}\mathbf{w} - \mathbf{y})$$

# %%
class LinearRegressionGD:
    """Linear Regression using Gradient Descent"""
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.intercept = None
        self.cost_history = []
        
    def compute_cost(self, X, y):
        """Compute mean squared error cost"""
        m = len(y)
        predictions = X @ self.weights + self.intercept
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        return cost
    
    def fit(self, X, y):
        """Train the model using gradient descent"""
        m, n = X.shape
        
        # Initialize parameters
        self.weights = np.random.normal(0, 0.01, n)
        self.intercept = 0
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            predictions = X @ self.weights + self.intercept
            
            # Compute cost
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1/m) * X.T @ (predictions - y)
            db = (1/m) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db
            
            # Check for convergence
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break
                
        print(f"Final cost: {cost:.2f}")
        return self
    
    def predict(self, X):
        """Make predictions"""
        return X @ self.weights + self.intercept

# Train gradient descent model on standardized data
print("Training Gradient Descent Linear Regression on Standardized Data...")
gd_lr = LinearRegressionGD(learning_rate=0.01, max_iterations=2000)
gd_lr.fit(X_train_scaled, y_train_scaled)

# Make predictions (in scaled space then transform back)
gd_pred_train_scaled = gd_lr.predict(X_train_scaled)
gd_pred_test_scaled = gd_lr.predict(X_test_scaled)

# Transform predictions back to original scale using our explicit function
gd_pred_train = inverse_standardize_target(gd_pred_train_scaled, target_mean, target_std)
gd_pred_test = inverse_standardize_target(gd_pred_test_scaled, target_mean, target_std)

# Calculate metrics in original scale
gd_train_mse = mean_squared_error(y_train, gd_pred_train)
gd_test_mse = mean_squared_error(y_test, gd_pred_test)
gd_train_r2 = r2_score(y_train, gd_pred_train)
gd_test_r2 = r2_score(y_test, gd_pred_test)

# Also calculate metrics in scaled space
gd_train_mse_scaled = mean_squared_error(y_train_scaled, gd_pred_train_scaled)
gd_test_mse_scaled = mean_squared_error(y_test_scaled, gd_pred_test_scaled)

print(f"Gradient Descent Results (Original Scale):")
print(f"  Train MSE: {gd_train_mse:.2f}")
print(f"  Test MSE:  {gd_test_mse:.2f}")
print(f"  Train R²:  {gd_train_r2:.4f}")
print(f"  Test R²:   {gd_test_r2:.4f}")

print(f"\nGradient Descent Results (Scaled Space):")
print(f"  Train MSE: {gd_train_mse_scaled:.6f}")
print(f"  Test MSE:  {gd_test_mse_scaled:.6f}")
print(f"  Coefficients (scaled): {gd_lr.weights}")
print(f"  Intercept (scaled): {gd_lr.intercept:.6f}")  # Should be ~0

# %% [markdown]
# ## Comprehensive Comparison of All Methods

# %%
# Create comparison summary
methods = ['Scikit-learn', 'Closed-Form', 'Gradient Descent']
train_mse = [sklearn_train_mse, closedform_train_mse, gd_train_mse]
test_mse = [sklearn_test_mse, closedform_test_mse, gd_test_mse]
train_r2 = [sklearn_train_r2, closedform_train_r2, gd_train_r2]
test_r2 = [sklearn_test_r2, closedform_test_r2, gd_test_r2]

comparison_df = pd.DataFrame({
    'Method': methods,
    'Train MSE': train_mse,
    'Test MSE': test_mse,
    'Train R²': train_r2,
    'Test R²': test_r2
})

print("Method Comparison:")
print("=" * 60)
print(comparison_df.round(4))

# Debug: Check if any values are abnormally large
print(f"\nDebugging - Check for scaling issues:")
print(f"Sklearn test MSE: {sklearn_test_mse:.2f}")
print(f"Closed-form test MSE: {closedform_test_mse:.2f}")
print(f"Gradient descent test MSE: {gd_test_mse:.2f}")

# Check prediction ranges
print(f"\nPrediction ranges (should be similar):")
print(f"Sklearn predictions: [{sklearn_pred_test.min():.2f}, {sklearn_pred_test.max():.2f}]")
print(f"Closed-form predictions: [{closedform_pred_test.min():.2f}, {closedform_pred_test.max():.2f}]")
print(f"Gradient descent predictions: [{gd_pred_test.min():.2f}, {gd_pred_test.max():.2f}]")
print(f"Actual values: [{y_test.min():.2f}, {y_test.max():.2f}]")

# Check if there are any NaN or infinite values
print(f"\nChecking for invalid values:")
print(f"Sklearn pred NaN: {np.isnan(sklearn_pred_test).sum()}, Inf: {np.isinf(sklearn_pred_test).sum()}")
print(f"Closed-form pred NaN: {np.isnan(closedform_pred_test).sum()}, Inf: {np.isinf(closedform_pred_test).sum()}")
print(f"Gradient descent pred NaN: {np.isnan(gd_pred_test).sum()}, Inf: {np.isinf(gd_pred_test).sum()}")

# %% [markdown]
# ## Visualizing the Comparison

# %%
# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Cost function for gradient descent
axes[0, 0].plot(gd_lr.cost_history)
axes[0, 0].set_title('Gradient Descent: Cost Function')
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Cost (MSE)')
axes[0, 0].grid(True, alpha=0.3)

# 2. Performance comparison
x_pos = np.arange(len(methods))
width = 0.35

axes[0, 1].bar(x_pos - width/2, test_mse, width, label='Test MSE', alpha=0.8)
axes[0, 1].bar(x_pos + width/2, train_mse, width, label='Train MSE', alpha=0.8)
axes[0, 1].set_xlabel('Method')
axes[0, 1].set_ylabel('MSE')
axes[0, 1].set_title('MSE Comparison')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(methods)
axes[0, 1].legend()

# 3. R² comparison
axes[0, 2].bar(x_pos - width/2, test_r2, width, label='Test R²', alpha=0.8)
axes[0, 2].bar(x_pos + width/2, train_r2, width, label='Train R²', alpha=0.8)
axes[0, 2].set_xlabel('Method')
axes[0, 2].set_ylabel('R² Score')
axes[0, 2].set_title('R² Score Comparison')
axes[0, 2].set_xticks(x_pos)
axes[0, 2].set_xticklabels(methods)
axes[0, 2].legend()

# 4. Predictions vs Actual (Test Set) - Scikit-learn
axes[1, 0].scatter(y_test, sklearn_pred_test, alpha=0.6)
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('Actual Values')
axes[1, 0].set_ylabel('Predicted Values')
axes[1, 0].set_title('Scikit-learn: Predictions vs Actual')
axes[1, 0].grid(True, alpha=0.3)

# 5. Predictions vs Actual (Test Set) - Closed Form
axes[1, 1].scatter(y_test, closedform_pred_test, alpha=0.6)
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 1].set_xlabel('Actual Values')
axes[1, 1].set_ylabel('Predicted Values')
axes[1, 1].set_title('Closed-Form: Predictions vs Actual')
axes[1, 1].grid(True, alpha=0.3)

# 6. Predictions vs Actual (Test Set) - Gradient Descent
axes[1, 2].scatter(y_test, gd_pred_test, alpha=0.6)
axes[1, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 2].set_xlabel('Actual Values')
axes[1, 2].set_ylabel('Predicted Values')
axes[1, 2].set_title('Gradient Descent: Predictions vs Actual')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Coefficient Comparison

# %%
# Compare the learned coefficients
coefficients_df = pd.DataFrame({
    'Feature': numeric_features,
    'Scikit-learn': sklearn_lr.coef_,
    'Closed-Form': closedform_lr.weights,
    'Gradient Descent': gd_lr.weights
})

print("Coefficient Comparison:")
print("=" * 50)
print(coefficients_df.round(6))

# Visualize coefficient comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Bar plot of coefficients
x_pos = np.arange(len(numeric_features))
width = 0.25

axes[0].bar(x_pos - width, sklearn_lr.coef_, width, label='Scikit-learn', alpha=0.8)
axes[0].bar(x_pos, closedform_lr.weights, width, label='Closed-Form', alpha=0.8)
axes[0].bar(x_pos + width, gd_lr.weights, width, label='Gradient Descent', alpha=0.8)
axes[0].set_xlabel('Features')
axes[0].set_ylabel('Coefficient Value')
axes[0].set_title('Coefficient Comparison Across Methods')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(numeric_features, rotation=45, ha='right')
axes[0].legend()

# Correlation between coefficients
axes[1].scatter(sklearn_lr.coef_, closedform_lr.weights, alpha=0.7, label='Sklearn vs Closed-Form')
axes[1].scatter(sklearn_lr.coef_, gd_lr.weights, alpha=0.7, label='Sklearn vs Gradient Descent')
axes[1].plot([sklearn_lr.coef_.min(), sklearn_lr.coef_.max()], 
             [sklearn_lr.coef_.min(), sklearn_lr.coef_.max()], 'r--', lw=2)
axes[1].set_xlabel('Scikit-learn Coefficients')
axes[1].set_ylabel('Other Method Coefficients')
axes[1].set_title('Coefficient Correlation')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate correlations
sklearn_vs_closed = np.corrcoef(sklearn_lr.coef_, closedform_lr.weights)[0, 1]
sklearn_vs_gd = np.corrcoef(sklearn_lr.coef_, gd_lr.weights)[0, 1]
closed_vs_gd = np.corrcoef(closedform_lr.weights, gd_lr.weights)[0, 1]

print(f"\nCoefficient Correlations:")
print(f"Scikit-learn vs Closed-Form: {sklearn_vs_closed:.6f}")
print(f"Scikit-learn vs Gradient Descent: {sklearn_vs_gd:.6f}")
print(f"Closed-Form vs Gradient Descent: {closed_vs_gd:.6f}")

# %% [markdown]
# ## Summary and Key Insights
#
# ### Performance Comparison:
# All three methods should produce very similar results, demonstrating that they're solving the same optimization problem.
# In general, you'd use the sci-kit learn one which internally will set up the linear regression equations and solve the linear equation defining beta. But, it's important to know the little details for when things go awry.

# %%
print("Analysis complete!")
print(f"\nFinal Comparison Summary:")
print(f"All methods achieved similar performance:")
for i, method in enumerate(methods):
    print(f"  {method:15}: Test R² = {test_r2[i]:.4f}, Test MSE = {test_mse[i]:.2f}")

print(f"\nThis demonstrates that all three approaches solve the same linear regression problem!")

# %%
