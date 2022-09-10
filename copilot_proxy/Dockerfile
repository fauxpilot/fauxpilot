# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip/ pip3 install -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip/ pip3 install --extra-index-url https://pypi.ngc.nvidia.com regex fire tritonclient[all]

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
