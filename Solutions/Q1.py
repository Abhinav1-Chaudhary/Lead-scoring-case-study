from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Assuming 'X' is your feature matrix and 'y' is the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Extract feature importances
feature_importances = model.coef_[0]

# Create a dictionary mapping feature names to their importance scores
feature_importance_dict = dict(zip(X.columns, feature_importances))

# Sort features based on importance
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)

# Print the top three variables
top_three_variables = sorted_features[:3]
print("Top Three Variables:")
for variable, importance in top_three_variables:
    print(f"{variable}: {importance}")
