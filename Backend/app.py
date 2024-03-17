from flask import Flask, jsonify, render_template, request
import joblib
import os
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict',methods=['POST','GET'])


def result():

    item_weight= float(request.form['item_weight'])
    item_fat_content=float(request.form['item_fat_content'])
    item_visibility= float(request.form['item_visibility'])
    item_type= float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year= float(request.form['outlet_establishment_year'])
    outlet_size= float(request.form['outlet_size'])
    outlet_location_type= float(request.form['outlet_location_type'])
    outlet_type= float(request.form['outlet_type'])

    X= np.array([[ item_weight,item_fat_content,item_visibility,item_type,item_mrp,
                  outlet_establishment_year,outlet_size,outlet_location_type,outlet_type ]])

    scaler_path= r'F:\Final _year\Backend\models\sc.sav'
    #scaler_path= r'Backend/models/XGBoost.sav'


    sc=joblib.load(scaler_path)
    #sc=load_model_from_github(scaler_path)

    X_std= sc.transform(X)

    #model_path=r'F:\Final _year\Backend\models\XGBoost.sav'
    model_path=r'F:\Final _year\Backend\models\XGBoost.sav'


    model= joblib.load(model_path)
    # model= load_model_from_github(model_path)

    Y_pred=model.predict(X_std)

    prediction_without_brackets = str(Y_pred[0])  # Assuming Y_pred is an array and you want the first element
    prediction_without_brackets = prediction_without_brackets.strip('[]')  # Remove the brackets
    prediction_without_brackets=int(float(prediction_without_brackets))
    return render_template("results.html", prediction = prediction_without_brackets)

if __name__ == '__main__':
    app.debug = True
    app.run()