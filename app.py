import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    if prediction == 0 :
        output = '<= 5 yrs'
    else : 
        output = '>= 5 yrs'

    return render_template('index.html', prediction_text="Player's career length is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)