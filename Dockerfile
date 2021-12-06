# Dockerfile ,Image ,Container

FROM python:3.7
ADD app.py .
WORKDIR ./
RUN pip install -r requirements.txt

CMD ["python", "app.py"]