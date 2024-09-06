FROM python:3.10-slim
WORKDIR /app
VOLUME /app/data
SHELL [ "/bin/bash", "-c" ]
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python3","/app/make_submission.py"]

COPY . /app

RUN pip install -r requirements.txt
RUN chmod +x /app/entrypoint.sh /app/baseline.py /app/make_submission.py
