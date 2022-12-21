from flask import Flask
from flask import request
from flask import jsonify
import numpy as np
from tensorflow import keras
from base64 import b64decode
from tensorflow.keras.preprocessing.image import load_img


model_file = 'model.h5'
target_size=(150, 150)
# loading the model

model = keras.models.load_model(model_file)

classes = ['Dark', 'Green',	'Light', 'Medium']



# def prepare_image(img, target_size):
#     img = img.resize(target_size, Image.Resampling.NEAREST)
#     return img

app = Flask('predict')


@app.route('/predict', methods=['POST'])
def predict():
    encoded_img = request.get_json()['im_data']
    decoded_data=b64decode(encoded_img)
#write the decoded data back to original format in  file
    with open('image.jpeg', 'wb') as img_file:
        img_file.write(decoded_data)

    x = load_img('image.jpeg', target_size=(150, 150))
    # img = keras.utils.get_file(origin=img_path)
    # print('image downloaded')
    img = np.array(x, dtype='float')/255 
    X = np.array([img])
    pred = model.predict(X)[0].astype('float')

    prediction = dict(zip(classes, np.around(pred, decimals=3)))
    
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9060)
