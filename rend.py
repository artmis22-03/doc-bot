import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Load the data from the CSV file
df = pd.read_csv('dataset.csv')  # Update with the actual path to your CSV file

# Fill missing values with a placeholder
df.fillna('none', inplace=True)

# Encode the target variable
label_encoder = LabelEncoder()
df['Disease'] = label_encoder.fit_transform(df['Disease'])

# Split the data into features and target
X = df.drop('Disease', axis=1)
y = df['Disease']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numeric and categorical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Random Forest': RandomForestClassifier(random_state=42)
}

for model_name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    # Cross-validation to assess the model
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"{model_name} - Cross-validation Accuracy: {np.mean(cv_scores):.2f}")
    
    # Train the model
    clf.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = clf.predict(X_test)
    
    # Print the classification report
    print(f"\nClassification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred))

# Function to classify a new test case
def classify_new_case(clf, new_case):
    # Create a DataFrame for the new case
    new_case_df = pd.DataFrame([new_case])
    
    # Predict the disease
    prediction = clf.predict(new_case_df)
    
    # Decode the prediction
    disease = label_encoder.inverse_transform(prediction)
    return disease[0]

# Example test case
new_case = {
    "Symptom_1": "fatigue",
    "Symptom_2": "weight_loss",
    "Symptom_3": "restlessness",
    "Symptom_4": "lethargy",
    "Symptom_5": "irregular_sugar_level",
    "Symptom_6": "none",
    "Symptom_7": "none",
    "Symptom_8": "none",
    "Symptom_9": "none",
    "Symptom_10": "none",
    "Symptom_11": "none",
    "Symptom_12": "none",
    "Symptom_13": "none",
    "Symptom_14": "none",
    "Symptom_15": "none",
    "Symptom_16": "none",
    "Symptom_17": "none"
}

# Classify the new test case using one of the trained models, e.g., Random Forest
clf = models['Random Forest']
clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
clf_pipeline.fit(X_train, y_train)  # Ensure the pipeline is trained
predicted_disease = classify_new_case(clf_pipeline, new_case)
with open('random_forest.pkl','wb') as file:
    pickle.dump(clf_pipeline, file)
print("\nPredicted Disease for the test case using Random Forest:", predicted_disease)
# Initialize lists to store training and testing accuracies
train_accuracies = []
test_accuracies = []

# Iterate over models
for model_name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    clf.fit(X_train, y_train)
    
    # Training accuracy
    train_acc = clf.score(X_train, y_train)
    train_accuracies.append(train_acc)
    
    # Testing accuracy
    test_acc = clf.score(X_test, y_test)
    test_accuracies.append(test_acc)

# Plotting the accuracies
plt.figure(figsize=(10, 6))
plt.bar(models.keys(), train_accuracies, label='Training Accuracy')
plt.bar(models.keys(), test_accuracies, label='Testing Accuracy')
plt.title('Training and Testing Accuracies of Models')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend()
plt.show()
# Predict on test set
y_pred = clf_pipeline.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
