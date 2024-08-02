import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load your datasets
train_data = pd.read_csv('C:/Users/AshokKumar/OneDrive/Desktop/Logistic/Titanic_Train.csv')
test_data = pd.read_csv('C:/Users/AshokKumar/OneDrive/Desktop/Logistic/Titanic_Test.csv')

# Preprocess the data (adjust based on your actual dataset)
# Drop rows with missing values in the target variable and the features of interest
train_data = train_data.dropna(subset=['Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Sex', 'Survived'])

# Features and target variable
X = train_data[['Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Sex']]
y = train_data['Survived']

# Encode 'Sex' column
X = pd.get_dummies(X, columns=['Sex'], drop_first=True)

# Drop the 'Ticket' column if it's not used in this case
# X = X.drop(columns=['Ticket'])

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the scaler and the model to files
joblib.dump(scaler, 'C:/Users/AshokKumar/OneDrive/Desktop/Logistic/scaler.pkl')
joblib.dump(model, 'C:/Users/AshokKumar/OneDrive/Desktop/Logistic/logistic_regression_model.pkl')

print("Model and scaler saved successfully.")
