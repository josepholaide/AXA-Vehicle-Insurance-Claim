# Install python 3.8
FROM python:3.7
RUN pip install xgboost pandas scikit-learn
ADD umoja.py /
 
ENTRYPOINT ["python", "-u", "/umoja.py"]
