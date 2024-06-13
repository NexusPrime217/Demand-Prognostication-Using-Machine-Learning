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
    								
    Store= float(request.form['Store'])
    Dept=10
    IsHoliday= float(request.form['IsHoliday'])
    Temperature= 32.6
    Fuel_Price = 103.38
    Type= float(request.form['Type'])
    Size= float(request.form['Size'])
    year= float(request.form['year'])
    month= float(request.form['month'])
    week= float(request.form['week'])
    item=int(request.form['item_type'])
    size=75000
    if Size==0:
        size=10000
    elif Size==1:
        size=50000   
    X= np.array([[ Store,Dept,IsHoliday,Temperature,Fuel_Price,Type,size,year,month ,week]])
    item_type=['Oil(1 ltr)','Sugar(1 kg)','Wheat(1 kg)','Rice(1 kg)','Soap','Ghee(1 Litre)','Cashew(250 gm)']
    
    

    scaler_path= r'F:\Final _year\Backend\models\sc.sav'
    


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
    return render_template("results.html", prediction = prediction_without_brackets,of_item=item_type[item])

if __name__ == '__main__':
    app.debug = True
    app.run()