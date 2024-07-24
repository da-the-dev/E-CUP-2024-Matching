FROM python:3.10
WORKDIR /app

COPY . /app

RUN mkdir -p /app/data

VOLUME /app/data
RUN pip3 install -r requirements.txt

RUN chmod +x /app/baseline.py
RUN chmod +x /app/make_submission.py
