FROM python:3.6.8

#ENV http_proxy http://70.10.15.10:8080
#ENV https_proxy http://70.10.15.10:8080
#RUN apt-get update && apt-get -o Acquire::BrokenProxy="true" -o Acquire::http::No-Cache="true" -o Acquire::http::Pipeline-Depth="0" install -y libgl1-mesa-glx
RUN apt-get update -o Acquire::http::proxy="http://70.10.15.10:8080" && apt-get -o Acquire::http::proxy="http://70.10.15.10:8080" -o Acquire::BrokenProxy="true" -o Acquire::http::No-Cache="true" -o Acquire::http::Pipeline-Depth="0" install -y libgl1-mesa-glx


ADD requirements.txt /
#RUN pip install --upgrade pip
#RUN pip install -r /requirements.txt
RUN pip install --proxy 70.10.15.10:8080 --upgrade pip
RUN pip install --proxy 70.10.15.10:8080 -r /requirements.txt

#ADD . /app
WORKDIR /app

#EXPOSE 5000
#CMD [ "python" , "app.py"]
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4" ]
