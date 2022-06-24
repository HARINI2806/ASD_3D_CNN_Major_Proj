#!/usr/bin/env python
# coding: utf-8

# In[1]:


from types import MethodType
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle


# In[2]:


app = Flask(__name__,template_folder='templates')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('about.html')

@app.route('/predict-get')
def predictget():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    return render_template('analyze.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    final_features = [np.array(features[0:10])]
    # final_features = np.squeeze(np.asarray(final_features))
    column_names=['A1_Score','A2_Score','A3_Score','A4_Score','A5_Score','A6_Score','A7_Score','A8_Score',
                      'A9_Score','A10_Score']
    print(final_features)
    final_features=pd.DataFrame(final_features,columns=column_names)
    print(final_features)
    prediction= model.predict(final_features)
    print(prediction)
    a=""
    if prediction[0]==0:
        a+="NO AUTISM"
    else:
        a+="AUTISM"
    return render_template('results.html', prediction_text='{}'.format(a))


if __name__ == '__main__':
	app.run()
