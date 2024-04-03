from flask import Flask, render_template, request
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import csv

app = Flask(__name__)

# Load data
training = pd.read_csv('Data/Training.csv')
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']

# Mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Train decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(x, y)

# Load additional data
severityDictionary = {}
description_list = {}
precautionDictionary = {}

def getSeverityDict():
    global severityDictionary
    with open('MasterData/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 2:  # Check if the row has at least 2 columns
                symptom = row[0]
                severity = row[1]
                severityDictionary[symptom] = int(severity)
            else:
                print(f"Error: Invalid row in symptom_severity.csv - {row}")


def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            description_list[row[0]] = row[1]

def getprecautionDict():
    global precautionDictionary
    with open('MasterData/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symptom_input = request.form['symptom_input']
        num_days = int(request.form['num_days'])
        
        severity = severityDictionary.get(symptom_input)
        if severity is None:
            error_message = f"Invalid symptom: {symptom_input}"
            return render_template('index.html', error_message=error_message)

        input_vector = np.zeros(len(cols))
        input_vector[cols.get_loc(symptom_input)] = 1

        prediction = clf.predict([input_vector])[0]
        predicted_disease = le.inverse_transform([prediction])[0]
        description = description_list.get(predicted_disease, "Description not available")
        precautions = precautionDictionary.get(predicted_disease, ["Precautions not available"])

        return render_template('index.html', output_message=f"You may have {predicted_disease}. {description}", precautions=precautions)
    else:
        symptoms = list(cols)
        return render_template('index.html', symptoms=symptoms)

if __name__ == '__main__':
    getSeverityDict()
    getDescription()
    getprecautionDict()
    app.run(debug=True)
