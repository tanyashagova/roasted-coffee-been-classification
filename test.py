import requests
import base64
# from PIL import Image
# from io import BytesIO


url = 'http://localhost:9060/predict'

image_path = 'green_been.jpg'


# Image data preparation #############
# img = Image.open(image_path)
# img = img.convert('L')   
# buffer = BytesIO()
# img.save(buffer, 'png')
# buffer.seek(0)
# data = buffer.read()
# data = base64.b64encode(data).decode()
# data = f'data:image/png;base64,{data}'
# print(data[:100])
# 
#
with open(image_path, "rb") as img_file:
    data = base64.b64encode(img_file.read())

image_data = {'im_data': data}

result = requests.post(url, json=image_data).json()
print(result)