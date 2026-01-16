from flask import Flask,render_template,request
import joblib
import pandas as pd
import numpy as np

app=Flask(__name__)

model=joblib.load('mall_model_v2.pkl')
scaler=joblib.load('mall_scaler_v2.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    age=int(request.form['age'])
    income=int(request.form['income'])
    score=int(request.form['score'])

    data=pd.DataFrame([[age,income,score]],columns=['Age','income','score'])
    scaled_data=scaler.transform(data)

    cluster=model.predict(scaled_data)[0]

    strategies = {
        0: {"name": "Sensible Seniors", "action": "Quality and comfort focused marketing."},
        1: {"name": "Elite VIPs", "action": "Priority service and exclusive event invites."},
        2: {"name": "Young Budgets", "action": "Social media trends and flash sales."},
        3: {"name": "Stable Millennial", "action": "Loyalty programs and cashback offers."},
        4: {"name": "The Disengaged Wealthy", "action": "Luxury re-engagement strategies."}
}
    
    answer=strategies.get(cluster)
    return render_template('index.html',cluster_name=answer['name'],action=answer['action'])

if __name__=='__main__':
    app.run(debug=True)

    
