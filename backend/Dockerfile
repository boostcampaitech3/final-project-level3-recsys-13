FROM python:3.8.5-slim-buster

ENV PYTHONUNBUFFERED 1

EXPOSE 30002
WORKDIR /app

COPY poetry.lock pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

COPY . ./
ENV PYTHONPATH app
ENTRYPOINT ["python", "app/main.py"]