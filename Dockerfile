FROM nvcr.io/nvidia/pytorch:22.06-py3

COPY requirements.txt requirements.txt
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install gfortran -y
RUN pip install -U pip
RUN pip install -r requirements.txt
