FROM python:3.7.3-alpine3.9
FROM jjanzic/docker-python3-opencv:opencv-4.0.1

LABEL maintainer="gaborpelesz@gmail.com"

WORKDIR /

COPY ./app /app

RUN pip3 install --upgrade pip
RUN pip3 install --trusted-host pypi.python.org -r /app/requirements.txt

CMD [ "python3", "/app/app.py" ]