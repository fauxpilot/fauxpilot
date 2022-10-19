FROM python:3.10-slim-buster

WORKDIR /python-docker

COPY copilot_proxy/requirements.txt requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

COPY copilot_proxy .

EXPOSE 5000

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "5000", "app:app"]
