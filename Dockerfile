FROM ubuntu:20.04

RUN apt update && apt -y upgrade
RUN apt -y install python3-pip
RUN pip3 install numpy tqdm matplotlib scipy

WORKDIR /usr/test

COPY ./data/emotions/ ./data/emotions/
COPY ./3d_visualiser.py ./3d_visualiser.py
COPY ./emotion_visualiser.py ./emotion_visualiser.py
COPY ./perlin.py ./perlin.py

RUN mkdir ./data/gifs

CMD [ "python3", "./3d_visualiser.py" ]
