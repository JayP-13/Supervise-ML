import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset from a CSV file
data = pd.read_csv('exampledataset1.csv')

# Step 2: Split the dataset into features and target variables
# Assuming the target variable is in the last column
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a logistic regression model using the training data
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate the model's performance by calculating the accuracy score on the testing data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Step 6: Print the accuracy score
print(f'Accuracy: {accuracy:.2f}')
