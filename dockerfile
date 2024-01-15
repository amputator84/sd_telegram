FROM python:3.11
LABEL maintainer="info@borisov-ab.ru"
COPY . /bot
WORKDIR /bot
RUN pip install -r requirements.txt
CMD ["python", "/bot/bot.py"]