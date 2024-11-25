from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
from diseases_list import diseases_list, diseases_list_cleaned
from symptoms_dict import symptoms_dict, symptoms_list


# flask app
app = Flask(__name__,template_folder='templates')


# load dataset
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")


# load model
svc = pickle.load(open('models/svc.pkl','rb'))

# custome and helping functions
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis] ['workout']


    return desc,pre,med,die,wrkout


# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

# creating routes

@app.route("/")
def index():
    return render_template("index.html",diseases_list=diseases_list_cleaned,symptoms_list=symptoms_list)

# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        if symptoms =="Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('index.html', message=message, diseases_list=diseases_list_cleaned,symptoms_list=symptoms_list)
        else:
            # Split the user's input into a list of symptoms (assuming they are comma-separated)
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            # Remove any extra characters, if any
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            if all(item in symptoms_list for item in user_symptoms):       
                predicted_disease = get_predicted_value(user_symptoms)
                dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
                my_precautions = []
                for i in precautions[0]:
                    my_precautions.append(i)
                return render_template('index.html',        
                predicted_disease=predicted_disease, dis_des=dis_des,
                my_precautions=my_precautions, 
                medications=medications, 
                my_diet=rec_diet,
                workout=workout,symptoms_list=symptoms_list,diseases_list=diseases_list_cleaned)

    return render_template('index.html',diseases_list=diseases_list_cleaned,symptoms_list=symptoms_list,user_input=symptoms)


if __name__ == '__main__':

    app.run(debug=True)