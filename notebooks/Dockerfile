FROM tensorflow/tensorflow:2.2.0-jupyter
LABEL maintainer="Zhong Li <Zhong.Li@intersystems.com>"

ENV http_proxy http://70.10.15.10:8080
ENV https_proxy http://70.10.15.10:8080
RUN apt-get update && apt-get install -y default-jre-headless && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --proxy 70.10.15.10:8080 --upgrade pip && \
    pip install --proxy 70.10.15.10:8080 --no-cache-dir -r requirements.txt

CMD /bin/bash -c "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token=''"