FROM python:3.9.2

ENV PYTHONUNBUFFERED 1

EXPOSE 8080
WORKDIR /app

COPY poetry.lock pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

RUN apt-get update \
  && apt-get -y install netcat gcc postgresql \
  && apt-get clean
RUN pip install -r requirements.txt

COPY . ./
ENV PYTHONPATH app
ENTRYPOINT ["python", "app/main.py"]