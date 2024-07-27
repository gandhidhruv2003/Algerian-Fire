import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

model = pickle.load(open("lin_reg.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        Temperature=float(request.form.get("Temperature"))
        RH=float(request.form.get("RH"))
        Ws=float(request.form.get("Ws"))
        Rain=float(request.form.get("Rain"))
        FFMC=float(request.form.get("FFMC"))
        DMC=float(request.form.get("DMC"))
        ISI=float(request.form.get("ISI"))
        Classes=float(request.form.get("Classes"))
        Region=float(request.form.get("Region"))
        
        new_data = scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = model.predict(new_data)
        return render_template("predict.html", results=result[0])
        
    else:
        return render_template("predict.html")

if __name__ == "__main__":
    app.run(debug=True)