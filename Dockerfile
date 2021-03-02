FROM python:3.6-slim-stretch

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev libglib2.0-0

WORKDIR /root
COPY . /app
RUN python3 -m pip install -r /app/requirements.txt

ENTRYPOINT ["python3", "/app/face_align.py"]