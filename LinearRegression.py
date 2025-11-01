import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib  # For saving the model

# a) Dataset Selection: Load MPG dataset from local CSV
df = pd.read_csv('auto-mpg.csv', na_values='?')

# b) Pre-processing
# Handle missing values: Drop rows with missing horsepower (6 rows)
df.dropna(subset=['horsepower'], inplace=True)

# Features and target
feature_cols = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year']
X = df[feature_cols].copy()
y = df['mpg']

# One-hot encode categorical 'origin' (1=USA, 2=Europe, 3=Japan)
origin_dummies = pd.get_dummies(df['origin'], prefix='origin', drop_first=True)
X = pd.concat([X, origin_dummies], axis=1)

# Standardize features (optional, but improves numerical stability)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# c) Model Training
# Split into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Linear Regression using scikit-learn
model = LinearRegression()
model.fit(X_train, y_train)

# Print coefficients and intercept
print("Regression Coefficients:")
for col, coef in zip(X.columns, model.coef_):
    print(f"{col}: {coef:.4f}")
print(f"\nIntercept: {model.intercept_:.4f}")

# Save the model
joblib.dump(model, 'lr_model.pkl')
print("\nModel saved as 'lr_model.pkl'")

# d) Evaluation
# Predict on test set
y_pred = model.predict(X_test)

# Calculate MSE, R², and MAE (as an additional accuracy metric)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nMean Squared Error (MSE): {mse:.4f}")
print(f"Accuracy (R²): {r2*100:.2f}%")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Save predictions to CSV
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_csv('predictions.csv', index=False)
print("\nPredictions saved as 'predictions.csv'")

# Plot Predicted vs Actual and save it
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title('Predicted vs Actual MPG')
plt.grid(True)
plt.savefig('predicted_vs_actual.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'predicted_vs_actual.png'")
plt.show()