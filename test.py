import requests
url = 'http://localhost:8080/predict'

# image = {'url': './data/test/Light/light (7).png'}
image = {'url': './green_been.jpg'}


result = requests.post(url, json=image).json()
print(result)