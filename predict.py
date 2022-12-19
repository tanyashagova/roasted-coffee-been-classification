from flask import Flask
from flask import request
from flask import jsonify
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img


model_file = 'model.h5'
target_size=(150, 150)
# loading the model

model = keras.models.load_model(model_file)

classes = ['Dark', 'Green',	'Light', 'Medium']


app = Flask('predict')


@app.route('/predict', methods=['POST'])
def predict():
    img_url = request.get_json()['url']
    img = load_img(img_url, target_size=target_size)
    img = np.array(img, dtype='float')/255 
    X = np.array([img])
    pred = model.predict(X)[0].astype('float')

    prediction = dict(zip(classes, np.around(pred, decimals=3)))
    
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
