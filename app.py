from flask import Flask, render_template, request
import jinja2
import pickle

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    model = pickle.load(open('model.pkl', 'rb'))
    vector = pickle.load(open('vect.pkl', 'rb'))

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = vector.transform(data).toarray() 
        #print(vect.shape)
        my_prediction = model.predict(vect)
    return render_template('home.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
