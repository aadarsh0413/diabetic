from flask import Flask, render_template, request, redirect
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('sc.pkl','rb'))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST","GET"])
def result():
    int_features = [float(i) for i in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(sc.transform(final_features))
    print(prediction)
    print(final_features)
    if prediction[0] == 0:
        return render_template("diabetic.html", txt=" non diabetic")
    else:
        return render_template("diabetic.html", txt="diabetic")





if __name__ == "__main__":
    app.run('0.0.0.0', debug=True)