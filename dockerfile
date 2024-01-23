FROM python:3.11
LABEL maintainer="info@borisov-ab.ru"

WORKDIR /bot
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "bot.py"]