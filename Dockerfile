FROM python:3.6

WORKDIR /fsl_osr

RUN apt update
RUN apt install libsndfile1

COPY requirements.txt .

RUN python3.6 -m pip install -r requirements.txt

