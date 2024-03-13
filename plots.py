import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_learning_curves(train_costs: list, test_costs: list):
    """
    Plot the learning curves for training and test costs.

    Args:
        train_costs (list): List of training costs.
        test_costs (list): List of test costs.
    """
    iterations = len(train_costs)

    plt.plot(range(1, iterations + 1), train_costs, 'k-', label='Train Cost')
    plt.plot(range(1, iterations + 1), test_costs, 'k--', label='Test Cost')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.legend()
    plt.title('Learning Curves')
    plt.show()


def plot_correlation_matrix(X, y):
    """
    Plot the correlation matrix between features and target variable.

    Args:
        X (pd.DataFrame): Features data.
        y (pd.Series): Target variable.
    """
    corr_matrix = X.join(y).corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='binary')
    plt.colorbar()
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.title('Correlation Matrix')
    plt.show()


def plot_pca_explained_variance_ratio(X_train, n_components=8):
    """
    Plot the cumulative explained variance ratio for the PCA components.

    Args:
        X_train (np.ndarray): Training features data.
        n_components (int, optional): Number of components to consider. Defaults to 9.
    """
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_), 'o-', color='black')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Explained Variance Ratio')
    plt.show()


def plot_pca_scatter(X_train_pca):
    """
    Plot the scatter plot of the first two principal components.

    Args:
        X_train_pca (np.ndarray): Transformed training data after applying PCA.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], marker='o', color='black', s=0.1)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Scatter Plot')
    plt.show()


def plot_pca_variance(X_train_pca):
    """
    Plot the bar chart of variances for the PCA components.

    Args:
        X_train_pca (np.ndarray): Transformed training data after applying PCA.
    """
    variances = np.var(X_train_pca, axis=0)
    num_components = len(variances)
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, num_components + 1), variances, color='black')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance')
    plt.title('PCA Component Variances')
    plt.show()
