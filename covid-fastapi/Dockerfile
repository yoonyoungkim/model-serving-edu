FROM python:3.6.8

ADD requirements.txt /
RUN pip install --proxy 70.10.15.10:8080 -r /requirements.txt

#ADD . /app
WORKDIR /app

#EXPOSE 5000
#CMD [ "python" , "app.py"]
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4" ]
