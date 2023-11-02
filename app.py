import pickle
import numpy as np
from flask import Flask,request,render_template

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    float_val = [float(a) for a in request.form.values()]
    final_val = np.array(float_val).reshape(1,6)
    predictions = model.predict(final_val)

    output = round(predictions[0],2)

    return render_template('index.html',results = 'The Insurance expenses would be $ {}'.format(output))

if __name__=="__main__":
    app.run(debug =True)