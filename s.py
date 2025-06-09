import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Advertising.csv")

# Drop unwanted index column
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# Preview dataset
print("📄 Dataset Preview:")
print(df.head())

# Check missing values
print("\n🔍 Missing Values:")
print(df.isnull().sum())

# Pairplot
sns.pairplot(df)
plt.suptitle("Pairplot of Advertising Dataset", y=1.02)
plt.show()
print("✅ Pairplot Done")

# Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
print("✅ Heatmap Done")

# Features and target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
print("✅ Model Trained")

# Predict
y_pred = model.predict(X_test)
print("✅ Prediction Done")

# Evaluation
print("\n📊 Model Evaluation:")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.3f}")
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")

# Coefficients
coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\n📈 Regression Coefficients:")
print(coef_df)

# Actual vs Predicted plot
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
print("✅ All Done")
