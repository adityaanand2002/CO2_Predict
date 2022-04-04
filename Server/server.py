from glob import glob
from unicodedata import name
import numpy as np
from flask import Flask, request,render_template,redirect
import pickle

app=Flask(__name__)
model=pickle.load(open('../model/model.pkl','rb'))

features=np.zeros(3)


@app.route("/",methods=["GET", "POST"])
def home():
    if request.method == 'POST':
        global features
        features[0]=request.form['engine_size']
        features[1]=request.form['cylinders']
        features[2]=request.form['fuel']
        print(features)
        return redirect("/predict")
    else:
        return render_template('index.html')

@app.route("/predict",methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        return redirect("/")
    else:
        prediction=model.predict([features])
        return  render_template('predict.html',prediction=prediction[0])
        

if __name__=='__main__':
    print("Starting Flask Server")
    app.run(debug=True)