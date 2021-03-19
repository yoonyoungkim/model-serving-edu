FROM python:3.6.8

#ENV http_proxy http://70.10.15.10:8080
#ENV https_proxy http://70.10.15.10:8080
RUN apt-get update && apt-get -o Acquire::BrokenProxy="true" -o Acquire::http::No-Cache="true" -o Acquire::http::Pipeline-Depth="0" install -y libgl1-mesa-glx
#RUN apt-get update -o Acquire::http::proxy="http://70.10.15.10:8080" && apt-get -o Acquire::http::proxy="http://70.10.15.10:8080" -o Acquire::BrokenProxy="true" -o Acquire::http::No-Cache="true" -o Acquire::http::Pipeline-Depth="0" install -y libgl1-mesa-glx

ADD requirements.txt /

#ADD imageio-2.9.0-py3-none-any.whl /
#ADD tensorflow-2.3.1-cp36-cp36m-manylinux2010_x86_64.whl /
#ADD numpy-1.18.4-cp36-cp36m-manylinux1_x86_64.whl /
#ADD opencv_python-4.4.0.46-cp36-cp36m-manylinux2014_x86_64.whl /
#ADD pandas-1.1.4-cp36-cp36m-manylinux1_x86_64.whl /
#ADD matplotlib-3.3.3-cp36-cp36m-manylinux1_x86_64.whl /
#ADD tensorboard-2.4.0-py3-none-any.whl /
#ADD grpcio-1.34.0-cp36-cp36m-manylinux2010_x86_64.whl /
#ADD Pillow-8.0.1-cp36-cp36m-manylinux1_x86_64.whl /
#ADD h5py-2.10.0-cp36-cp36m-manylinux1_x86_64.whl /
#ADD scikit_image-0.17.2-cp36-cp36m-manylinux1_x86_64.whl /
#ADD scipy-1.5.4-cp36-cp36m-manylinux1_x86_64.whl /
#ADD PyWavelets-1.1.1-cp36-cp36m-manylinux1_x86_64.whl /

#RUN pip install --proxy 70.10.15.10:8080 --upgrade pip

#RUN pip install --proxy 70.10.15.10:8080 wheel
#RUN pip install --proxy 70.10.15.10:8080 numpy-1.18.4-cp36-cp36m-manylinux1_x86_64.whl
#RUN pip install --proxy 70.10.15.10:8080 pandas-1.1.4-cp36-cp36m-manylinux1_x86_64.whl
#RUN pip install --proxy 70.10.15.10:8080 Pillow-8.0.1-cp36-cp36m-manylinux1_x86_64.whl
#RUN pip install --proxy 70.10.15.10:8080 imageio-2.9.0-py3-none-any.whl
#RUN pip install --proxy 70.10.15.10:8080 matplotlib-3.3.3-cp36-cp36m-manylinux1_x86_64.whl
#RUN pip install --proxy 70.10.15.10:8080 grpcio-1.34.0-cp36-cp36m-manylinux2010_x86_64.whl
#RUN pip install --proxy 70.10.15.10:8080 tensorboard-2.4.0-py3-none-any.whl
#RUN pip install --proxy 70.10.15.10:8080 h5py-2.10.0-cp36-cp36m-manylinux1_x86_64.whl 
#RUN pip install --proxy 70.10.15.10:8080 tensorflow-2.3.1-cp36-cp36m-manylinux2010_x86_64.whl 
#RUN pip install --proxy 70.10.15.10:8080 opencv_python-4.4.0.46-cp36-cp36m-manylinux2014_x86_64.whl
#RUN pip install --proxy 70.10.15.10:8080 scipy-1.5.4-cp36-cp36m-manylinux1_x86_64.whl
#RUN pip install --proxy 70.10.15.10:8080 PyWavelets-1.1.1-cp36-cp36m-manylinux1_x86_64.whl
#RUN pip install --proxy 70.10.15.10:8080 scikit_image-0.17.2-cp36-cp36m-manylinux1_x86_64.whl

#RUN pip install --proxy 70.10.15.10:8080 -r /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /requirements.txt

#ADD . /app
WORKDIR /app

EXPOSE 5000
#CMD [ "python" , "app.py", "0.0.0.0:5000"]
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "2"]