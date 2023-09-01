FROM anibali/pytorch:1.13.0-cuda11.8

WORKDIR /DeepTrader
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN sudo apt-get update
RUN sudo apt-get -y install make
EXPOSE 8889
ENTRYPOINT ["jupyter", "notebook","--ip=0.0.0.0","--no-browser", "--port=8889", "--NotebookApp.token=''", "--NotebookApp.password=''"]