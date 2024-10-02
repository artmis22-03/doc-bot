import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the data from the CSV file
df = pd.read_csv('dataset.csv')  # Update with the actual path to your CSV file

# Fill missing values with a placeholder
df.fillna('none', inplace=True)

# Encode the categorical variables
label_encoders = {}
for column in df.columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Split the data into features and target
X = df.drop('Disease', axis=1)
y = df['Disease']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Save the decision tree classifier to a pkl file
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(clf, file)

# Initialize and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Save the KNN classifier to a pkl file
with open('knn_model.pkl', 'wb') as file:
    pickle.dump(knn, file)

# Predict on the test set using the decision tree classifier
y_pred_tree = clf.predict(X_test)
# Generate the classification report for the decision tree classifier
report_tree = classification_report(y_test, y_pred_tree, target_names=label_encoders['Disease'].classes_)
print("\nDecision Tree Classification Report:\n")
print(report_tree)

# Predict on the test set using the KNN classifier
y_pred_knn = knn.predict(X_test)
# Generate the classification report for the KNN classifier
report_knn = classification_report(y_test, y_pred_knn, target_names=label_encoders['Disease'].classes_)
print("\nKNN Classification Report:\n")
print(report_knn)

# Function to classify a new test case
def classify_new_case(model, label_encoders, new_case):
    # Create a DataFrame for the new case
    new_case_df = pd.DataFrame([new_case])
    
    # Encode the new case using the same encoders
    for column in new_case_df.columns:
        le = label_encoders[column]
        # Add new unseen labels to the classes
        unseen_labels = set(new_case_df[column].unique()) - set(le.classes_)
        if unseen_labels:
            le.classes_ = np.append(le.classes_, list(unseen_labels))
        new_case_df[column] = le.transform(new_case_df[column])
    
    # Predict the disease
    prediction = model.predict(new_case_df)
    
    # Decode the prediction
    disease = label_encoders['Disease'].inverse_transform(prediction)
    return disease[0]

# Example test case
new_case = {
    "Symptom_1": "chills",
    "Symptom_2": "joint_pain",
    "Symptom_3": "vomiting",
    "Symptom_4": "fatigue",
    "Symptom_5": "high_fever",
    "Symptom_6": "headache",
    "Symptom_7": "nausea",
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

# Classify the new test case using the decision tree classifier
predicted_disease_tree = classify_new_case(clf, label_encoders, new_case)
print("\nPredicted Disease for the test case using Decision Tree:", predicted_disease_tree)

# Classify the new test case using the KNN classifier
predicted_disease_knn = classify_new_case(knn, label_encoders, new_case)
print("\nPredicted Disease for the test case using KNN:", predicted_disease_knn)

# Load the symptom description data
df1 = pd.read_csv("symptom_Description.csv")
dfdict = df1.set_index('Disease').T.to_dict('list')

# Print the description of the predicted disease using the decision tree classifier
print("\nSymptom description for predicted disease (Decision Tree):")
print(dfdict[predicted_disease_tree][0])

# Print the description of the predicted disease using the KNN classifier
print("\nSymptom description for predicted disease (KNN):")
print(dfdict[predicted_disease_knn][0])

# Debug: Verify the encoding of the new case
encoded_new_case = {}
for column in new_case:
    encoded_new_case[column] = label_encoders[column].transform([new_case[column]])[0]
print("\nEncoded New Case:", encoded_new_case)
