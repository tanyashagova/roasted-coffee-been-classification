This repository contains the capstone progect for ml-zoomcamp-2022 course.

The aim of the project is to solve the task of coffee beans classification according to the raosting level. 

This project uses [Cofedataset of employees attrition](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224). You can download the dataset directly from [Kaggle](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224) or from the [data folder] in the progect on GitHub.


## Description

Project folder contains

* Data 
* Notebook (`notebook.ipynb`) with data preparation, EDA, and model selection process
* Script `train.py` which contains  training and saving the final model
* File `model.h5` with final model
* Script `predict.py` with model loading and serving it via a web serice (with Flask)
* `Pipenv` and `Pipenv.lock` files with dependencies
* `Dockerfile` for running the service
* Script `test.py` with test prediction for data of given person 


## Usage

Clone the repository of the project on your computer. Then

1. Install dependencies from Pipfile by running command:
```sh
pipenv install
```
2. Activate virtual environment:
```sh
pipenv shell
```
3. Run service with waitress:
```sh
waitress-serve --listen=0.0.0.0:9060 predict:app
```

4. Run test.py to see attrition prediction on given data.

Alternatively you can run service with Docker:
1. Build an image from a Dockerfile by running following command:
```sh
docker build -t roast-coffee-bean .
```
2. Run service:
```sh
docker run --rm -it -p 9060:9060 -d  roast-coffee-bean
```
3. Run test.py to see probabilities of the picture belonging to spesific class.

After that you will see following:

![Result_image](screen.png)

4. Try to change different parameters in record from test.py and see attrition result.