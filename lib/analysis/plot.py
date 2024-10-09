import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.integrate import cumtrapz
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression


def plot_error_density(best_model_errors, test_size):
    errors = best_model_errors[test_size]
    plt.figure(figsize=(8, 6))
    sns.kdeplot(errors, fill=True, color='skyblue', label=f'Error difference of the best model for test size = {test_size}')
    plt.hist(errors, bins=30, density=True, alpha=0.3, color='blue')
    plt.title(f'Probability density of errors of the best model for test_size={test_size}')
    plt.xlabel('Error difference')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'error_density_test_size_{test_size}.png')
    plt.show()


def get_quantile_error(errors, alpha):
    kde = gaussian_kde(errors)
    error_range = np.linspace(min(errors), max(errors), 1000)
    kde_values = kde(error_range)
    cdf = cumtrapz(kde_values, error_range, initial=0)
    cdf /= cdf[-1]
    quantile_value = error_range[np.searchsorted(cdf, 1 - alpha)]
    return quantile_value


def plot_max_error_for_alpha_kde(best_model_errors, alpha):
    test_sizes = sorted(best_model_errors.keys())
    max_errors = []

    for test_size in test_sizes:
        errors = best_model_errors[test_size]
        quantile_value = get_quantile_error(errors, alpha)
        max_errors.append(quantile_value)
    plt.figure(figsize=(10, 6))
    plt.plot(test_sizes, max_errors, marker='o', color='blue', label=f'Maximum error difference for alpha={alpha}')
    plt.title(f'Maximum error difference based on KDE corresponding to alpha={alpha} for different test sizes')
    plt.xlabel('Test Size')
    plt.ylabel('Maximum error difference')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'max_error_for_alpha_{alpha}_kde.png')
    plt.show()


def plot_log_error_quantiles_and_mean_with_regression(best_model_errors, alphas):
    test_sizes = sorted(best_model_errors.keys())
    log_test_sizes = np.log(test_sizes)
    mean_errors = []
    plt.figure(figsize=(12, 8))

    for alpha in alphas:
        log_quantile_errors = []
        for test_size in test_sizes:
            errors = best_model_errors[test_size]
            quantile_value = get_quantile_error(errors, alpha)
            log_quantile_errors.append(np.log(quantile_value))

        plt.plot(log_test_sizes, log_quantile_errors, marker='o', label=f'Log(Quantile) for alpha={alpha}')

    for test_size in test_sizes:
        errors = best_model_errors[test_size]
        mean_value = np.mean(errors)
        mean_errors.append(np.log(mean_value))

    plt.plot(log_test_sizes, mean_errors, marker='x', color='black', linestyle='--', label='Log(Mean errors difference)')

    # Linear Regression
    X = log_test_sizes.reshape(-1, 1)  # Reshape for regression
    y = mean_errors
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    slope = model.coef_[0]
    intercept = model.intercept_
    plt.plot(log_test_sizes, y_pred, color='red', label=f'Regression Line (slope={slope:.4f})')

    plt.title('Logarithm of quantiles and mean errors difference with linear regression (KDE)')
    plt.xlabel('Log(Test Size)')
    plt.ylabel('Log(Error difference)')
    plt.grid(True)
    plt.legend()
    plt.savefig('log_error_quantiles_and_mean_with_regression.png')
    plt.show()


def plot_error_quantiles_and_mean(best_model_errors, alphas):
    test_sizes = sorted(best_model_errors.keys())
    mean_errors = []
    plt.figure(figsize=(12, 8))

    for alpha in alphas:
        quantile_errors = []
        for test_size in test_sizes:
            errors = best_model_errors[test_size]
            quantile_value = get_quantile_error(errors, alpha)
            quantile_errors.append(quantile_value)

        plt.plot(test_sizes, quantile_errors, marker='o', label=f'Quantile (1-alpha) for alpha={alpha}')

    for test_size in test_sizes:
        errors = best_model_errors[test_size]
        mean_value = np.mean(errors)
        mean_errors.append(mean_value)

    plt.plot(test_sizes, mean_errors, marker='x', color='black', linestyle='--', label='Mean Errors')

    plt.title('Quantiles of Errors difference and Mean errors difference for Different Test Sizes')
    plt.xlabel('Test Size')
    plt.ylabel('Error difference')
    plt.grid(True)
    plt.legend()
    plt.savefig('error_quantiles_and_mean.png')
    plt.show()
