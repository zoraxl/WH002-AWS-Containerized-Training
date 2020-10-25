FROM python:3.7

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY code/train.py /opt/ml/code/

ENTRYPOINT ["python", "/opt/ml/code/train.py"]