# neural-network-challenge-1

# Student Loan Risk with Deep Learning

# Imports
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from pathlib import Path

# Read the csv into a Pandas DataFrame
file_path = "https://static.bc-edx.com/ai/ail-v-1-0/m18/lms/datasets/student-loans.csv"
loans_df = pd.read_csv(file_path)

# Review the DataFrame
loans_df.head()

# Review the data types associated with the columns
loans_df.dtypes

# Check the credit_ranking value counts
loans_df["credit_ranking"].value_counts()

# Define the target set y using the credit_ranking column
y = loans_df["credit_ranking"]

# Display a sample of y
y[:5]

# Define features set X by selecting all columns but credit_ranking
x = loans_df.drop(columns=["credit_ranking"])

# Review the features DataFrame
x.head()

from re import X
# Split the preprocessed data into a training and testing dataset
# Assign the function a random_state equal to 1
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)

# Create a StandardScaler instance
x_scaler = StandardScaler()

# Fit the scaler to the features training dataset
x_scaler.fit(X_train)

# Fit the scaler to the features training dataset
X_train_scaled = x_scaler.transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

# Define the the number of inputs (features) to the model
input_nodes = len(x.columns)

# Review the number of features
input_nodes

# Define the number of hidden nodes for the first hidden layer
hidden_nodes_layer1 =  28

# Define the number of hidden nodes for the second hidden layer
hidden_nodes_layer2 = 14


# Define the number of neurons in the output layer
output_nodes = 1

# Create the Sequential model instance
nn_model = tf.keras.models.Sequential()

# Add the first hidden layer
nn_model.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, activation="relu", input_dim=input_nodes))

# Add the second hidden layer
nn_model.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

# Add the output layer to the model specifying the number of output neurons and activation function
nn_model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Display the Sequential model summary
nn_model.summary()

# Compile the Sequential model
nn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Fit the model using 50 epochs and the training data
fit_model = nn_model.fit(X_train_scaled, y_train, epochs=50)

# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
model_loss, model_accuracy = nn_model.evaluate(X_test_scaled,y_test,verbose=2)


# Display the model loss and accuracy results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# Set the model's file path
file_path = Path("student_loans.keras")

# Export your model to a keras file
nn_model.save(file_path)

# Set the model's file path
file_path = Path("student_loans.keras")

# Load the model to a new object
nn_imported = tf.keras.models.load_model(file_path)

# Make predictions with the test data
predictions = nn_imported.predict(X_test_scaled,verbose=2)

# Display a sample of the predictions
predictions[:5]

# Save the predictions to a DataFrame and round the predictions to binary results
predictions_df = pd.DataFrame(columns=["predictions"], data=predictions)
predictions_df["predictions"] = round(predictions_df["predictions"],0)
predictions_df

# Print the classification report with the y test data and predictions
print(classification_report(y_test, predictions_df["predictions"].values))

Briefly answer the following questions in the space provided:

1. Describe the data that you would need to collect to build a recommendation system to recommend student loan options for students. Explain why this data would be relevant and appropriate.
  In order to build a student loan recommendation system, I would use the financial aid score, locaton parameter, payment history, total loan score, time to completion, major and gpa ranking.  This information is helpful, because it gives a guideline for how the applicant handles payment by both the applicants past history and those who are close in profile to the applicant.

2. Based on the data you chose to use in this recommendation system, would your model be using collaborative filtering, content-based filtering, or context-based filtering? Justify why the data you selected would be suitable for your choice of filtering method.
  Using all of the data in the dataframe uses context-based filtering.  The data I selected would give information about the individual applicant as well as other applicants who have similiar qualities/parameters as the applicant.

3. Describe two real-world challenges that you would take into consideration while building a recommendation system for student loans. Explain why these challenges would be of concern for a student loan recommendation system.
  The real world challenges are that no two individuals are alike and every persons circumstances aren't alike.  This does not account for all of the outliers. A student who selects a certain major, requires certain financial aid or has a certain GPA may end up being more responsible or capable of repayment than others who have the same major or GPA. There may be certain changes in an individuals life than may affect the outcome and alter their ability to desire to make repayment.