from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
app.debug = True
with open('nb_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ''
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([features])
        prediction_text = "Malignant" if prediction[0] == 4 else "Benign"
    return render_template('index.html', prediction_text=prediction_text)



if __name__ == '__main__':
    app.run(debug=True)

