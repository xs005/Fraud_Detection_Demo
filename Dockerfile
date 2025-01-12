FROM python:3.10-slim

WORKDIR /app
ADD . /app

RUN pip install -r requirements.txt

RUN ["chmod", "+x", "/app/run_api.sh"]
ENTRYPOINT ["bash", "/app/run_api.sh"]