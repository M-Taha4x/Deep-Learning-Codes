import numpy as np

# Generate noisy data from a simple true pattern: y = 2x + 1 + noise
np.random.seed(42)
x_train = np.linspace(0, 10, 15)
y_true = 2 * x_train + 1
noise = np.random.normal(0, 3, size=x_train.shape)
y_train = y_true + noise

# Fit polynomials of increasing degree (increasing "model complexity")
degrees = [1, 3, 14]   # 1=underfit-ish, 3=reasonable, 14=severe overfit
for d in degrees:
    coeffs = np.polyfit(x_train, y_train, d)
    y_pred_train = np.polyval(coeffs, x_train)
    train_error = np.mean((y_train - y_pred_train) ** 2)
    
    # Test on NEW unseen points from the same true pattern
    x_test = np.linspace(0, 10, 15) + 0.3   # slightly shifted, "unseen" points
    y_test_true = 2 * x_test + 1 + np.random.normal(0, 3, size=x_test.shape)
    y_pred_test = np.polyval(coeffs, x_test)
    test_error = np.mean((y_test_true - y_pred_test) ** 2)
    
    print(f"Degree {d}: Train MSE={train_error:.2f}, Test MSE={test_error:.2f}, Gap={test_error-train_error:.2f}")