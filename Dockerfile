# Install python 3.8
FROM python:3.8
RUN pip install xgboost
ADD umoja.py /
 
ENTRYPOINT ["python", "-u", "/umoja.py"]
