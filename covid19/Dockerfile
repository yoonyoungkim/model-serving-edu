FROM python:3.6.8

ENV http_proxy http://70.10.15.10:8080
ENV https_proxy http://70.10.15.10:8080
RUN apt-get update && apt-get install -y libgl1-mesa-glx

ADD requirements.txt /
RUN pip install --proxy 70.10.15.10:8080 --upgrade pip
RUN pip install --proxy 70.10.15.10:8080 -r /requirements.txt

#ADD . /app
WORKDIR /app

EXPOSE 5000
#CMD [ "python" , "app.py"]
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "2"]
