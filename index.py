import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# 1. Calculate the Correlation Matrix
# This measures how much each variable moves with the others (1.0 = perfect match, -1.0 = opposite)
df=pd.read_csv('data.csv')
corr_matrix = df.corr()

# 2. Plot the Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Refinery Process Correlation Heatmap")
plt.savefig('Figure_1.png', dpi=1200)

# 3. Check specific correlations with Yield
print("Correlation with Butane_Yield:")
print(corr_matrix['Butane_Yield'].sort_values(ascending=False))

from sklearn.model_selection import train_test_split

# 1. Drop the target to define Inputs (X)
# We also drop 'Bottom_Temp_Red' if your heatmap showed it was 100% redundant
X = df.drop(columns=['Butane_Yield', 'Bottom_Temp_Red'])

# 2. Define Target (y)
y = df['Butane_Yield']

# 3. Split into dfing (80%) and Testing (20%) sets
# random_state=42 ensures we get the same split every time (reproducibility)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ Data Split Successfully")
print(f"dfing Data Shape: {X_train.shape}")
print(f"Testing Data Shape: {X_test.shape}")



from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Train Linear Regression (The Baseline)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict
y_pred_lr = lr_model.predict(X_test)

# Evaluate
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("📊 Linear Regression Results:")
print(f"MAE: {mae_lr:.4f} (Average Error)")
print(f"R²:  {r2_lr:.4f} (Fit Quality)")
print("-" * 30)

# 2. Train Random Forest (The Champion)
# n_estimators=100 means we use 100 decision trees
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test)

# Evaluate
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("🌲 Random Forest Results:")
print(f"MAE: {mae_rf:.4f}")
print(f"R²:  {r2_rf:.4f}")


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- 1. Feature Importance ---
# This asks the model: "Which column helped you make the best predictions?"
importance = rf_model.feature_importances_
feature_names = X_train.columns

# Create a DataFrame for plotting
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
fi_df = fi_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
plt.title('Feature Importance: What Drives Refinery Yield?')
plt.xlabel('Relative Importance (0-1)')
plt.ylabel('Process Variable')
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
# plt.show()

# --- 2. Predicted vs. Actual (Model Validation) ---
# A perfect model would have all dots on the red diagonal line
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_rf, alpha=0.5, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
plt.xlabel('Actual Yield (from Dataset)')
plt.ylabel('Predicted Yield (by Random Forest)')
plt.title('Model Accuracy: Predicted vs. Actual')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()

# Print the top driver
top_feature = fi_df.iloc[0]['Feature']
print(f"✅ Analysis Complete.")
print(f"The most critical process variable is: {top_feature}")

import numpy as np
import plotly.graph_objects as go

# 1. Define the Range for our Grid
# We want to sweep across the Min and Max of our two key variables
x_range = np.linspace(X_train['Reflux_Flow'].min(), X_train['Reflux_Flow'].max(), 50)
y_range = np.linspace(X_train['Top_Temp'].min(), X_train['Top_Temp'].max(), 50)

# 2. Create a Meshgrid (a 50x50 grid of points)
x_grid, y_grid = np.meshgrid(x_range, y_range)

# 3. Prepare the Input for Prediction
# We need to hold the other variables constant (at their mean values) to isolate Reflux & Temp
mean_values = X_train.mean()

# Create a placeholder DataFrame for the grid
grid_data = pd.DataFrame(index=range(x_grid.size), columns=X_train.columns)

# Fill every row with the average values
for col in X_train.columns:
    grid_data[col] = mean_values[col]

# Overwrite the Reflux and Temp columns with our grid values
grid_data['Reflux_Flow'] = x_grid.ravel()
grid_data['Top_Temp'] = y_grid.ravel()

# 4. Predict Yield for every point on the grid
z_grid = rf_model.predict(grid_data)
z_grid = z_grid.reshape(x_grid.shape)

# 5. Plot Interactive 3D Surface
fig = go.Figure(data=[go.Surface(
    z=z_grid,
    x=x_grid,
    y=y_grid,
    colorscale='Viridis',
    colorbar=dict(title='Butane Content (Yield)')
)])

fig.update_layout(
    title='Optimization Surface: minimizing Butane Impurity',
    scene=dict(
        xaxis_title='Reflux Flow',
        yaxis_title='Top Temp',
        zaxis_title='Predicted Butane Yield'
    ),
    width=800,
    height=600
)

fig.show()


