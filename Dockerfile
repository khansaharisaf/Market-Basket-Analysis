FROM python:3.7-slim

ENV TZ=Asia/Jakarta
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY ./requirements.txt requirements.txt
RUN apt-get update && apt-get install -y libpq-dev gcc
RUN pip install -r requirements.txt

COPY ./app /app
WORKDIR /app

#streamlit config
RUN mkdir -p /root/.streamlit
COPY ./config.toml /root/.streamlit/config.toml

#PORTING TO 11000:8501
EXPOSE 8501
CMD ["streamlit", "run", "main.py"]

#CMD tail -f /dev/null
