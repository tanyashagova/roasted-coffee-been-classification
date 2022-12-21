FROM python:3.9-slim


RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv lock
RUN pipenv install --system --deploy

COPY ["predict.py", "model.h5", "./"]

EXPOSE 9060

ENTRYPOINT ["waitress-serve"]
CMD ["--port=9060", "predict:app"]