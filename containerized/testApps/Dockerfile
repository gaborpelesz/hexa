FROM python:3.7.3-alpine3.9
FROM jjanzic/docker-python3-opencv:opencv-4.0.1

WORKDIR /app

COPY ./app /app

RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 80

ADD ./app/images /images

CMD [ "python3", "./app.py" ]