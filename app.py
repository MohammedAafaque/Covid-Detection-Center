from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template('test.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    print(request.form)
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    prediction = model.predict_proba(final)
    result = model.predict(final)
    if result == 1:
        return render_template('positive.html', pred='You are Covid Positive.')
    else:
        return render_template('negative.html', pred='You are Covid Negative.')


if __name__ == "__main__":
    app.run(debug=True)

