FROM python:3.7

WORKDIR /app

COPY streamlit_app.py streamlit_app.py
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8501

COPY . /app

ENTRYPOINT ["streamlit", "run"]

CMD ["streamlit_app.py"]
