import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset (Replace this with your dataset path)
df = pd.read_csv('train.csv')

# Select relevant columns (features)
features = df[['GrLivArea', 'TotRmsAbvGrd', 'OverallQual']]  # You can add more features
target = df['SalePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(model, 'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Print model evaluation
print(f'Model R-squared: {model.score(X_test_scaled, y_test)}')
