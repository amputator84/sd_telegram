FROM python:3.11
LABEL maintainer="info@borisov-ab.ru"

COPY requirements.txt /bot
WORKDIR /bot
RUN pip install -r requirements.txt

COPY . /bot
CMD ["python", "/bot/bot.py"]