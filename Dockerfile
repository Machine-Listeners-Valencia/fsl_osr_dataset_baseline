FROM python:3.6

WORKDIR /fsl_osr

COPY requirements.txt .

RUN python3.6 -m pip install -r requirements.txt

