FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Treina o modelo durante o build (opcional, ou você pode copiar os arquivos .h5 se já tiver)
# RUN python train_model.py 

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]