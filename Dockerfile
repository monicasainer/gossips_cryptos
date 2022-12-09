FROM python:3.10.6-slim
WORKDIR /prod
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY gossips_cryptos gossips_cryptos
CMD uvicorn gossips_cryptos.api.fast:app --host 0.0.0.0 --port $PORT
