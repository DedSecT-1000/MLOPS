import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data  = pd.read_csv("C:/Users/Chengappa/Desktop/MLOPS-1/Practicals/wine-quality.csv")
print(data)

X = data.drop('quality', axis=1)
y = data['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model using the training sets
model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = model.predict(X_test)

# Print the coefficients
print('Coefficients:', model.coef_)

# Print the mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))

# Print the coefficient of determination (R^2)
print('Coefficient of determination (R^2): %.2f' % r2_score(y_test, y_pred))
