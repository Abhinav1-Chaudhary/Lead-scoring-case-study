from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Assuming 'X' is your feature matrix, and 'y' is the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical features
categorical_features = X.select_dtypes(include=['object']).columns

# Create a column transformer to handle categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Create a pipeline with preprocessing and logistic regression
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression())])

# Train the model
model.fit(X_train, y_train)

# Extract feature importances after one-hot encoding
feature_importances = model.named_steps['classifier'].coef_[0]

# Create a dictionary mapping feature names to their importance scores
feature_importance_dict = dict(zip(model.named_steps['preprocessor'].get_feature_names_out(input_features=categorical_features), feature_importances))

# Sort features based on importance
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)

# Print the top three categorical variables
top_three_categorical_variables = sorted_features[:3]
print("Top Three Categorical Variables:")
for variable, importance in top_three_categorical_variables:    print(f"{variable}: {importance}")

