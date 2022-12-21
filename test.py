import requests
import base64

url = 'http://localhost:9060/predict'

image_path = 'green_been.jpg'

with open(image_path, "rb") as img_file:
    data = base64.b64encode(img_file.read())


image_data = {'im_data': data}


result = requests.post(url, json=image_data).json()
print(result)