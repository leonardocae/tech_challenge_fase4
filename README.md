🧠 Tech Challenge Fase 4 – Previsão de Ações com LSTM

Este projeto implementa um pipeline completo de Machine Learning para prever preços de ações utilizando uma rede neural LSTM.
Inclui:

Coleta automática de dados do Yahoo Finance

Pré-processamento

Treinamento e salvamento do modelo

Deploy via API usando FastAPI

(Opcional) Deploy em container Docker

📁 Estrutura do Projeto
tech_challenge_fase4/
│
├── app.py                # Código da API FastAPI
├── train_model.py        # Script de treinamento do modelo
├── requirements.txt      # Dependências do projeto
├── Dockerfile            # (Opcional) Para rodar via Docker
├── README.md             # Documentação
└── .gitignore            # Arquivos ignorados no Git

🧩 Funcionalidades
✔ Treinamento do modelo

Baixa os dados históricos da ação PETR4 (pode ser modificado)

Normaliza os dados com MinMaxScaler

Cria sequências de 60 dias

Treina uma LSTM com Dropout

Salva:

lstm_stock_model.h5

scaler.pkl

✔ API FastAPI

Endpoints disponíveis:

Método	Rota	Descrição
GET	/	Status da API
POST	/predict	Recebe os últimos 60 preços e retorna a previsão
⚙️ Como Executar o Projeto
📌 1. Instalar Dependências

No terminal:

pip install -r requirements.txt

📌 2. Treinar o Modelo

Execute:

python train_model.py


Após o treinamento, serão gerados:

lstm_stock_model.h5
scaler.pkl


Esses arquivos são obrigatórios para a API funcionar.

📌 3. Rodar a API

Com os arquivos do modelo disponíveis, execute:

uvicorn app:app --reload


A API estará disponível em:

http://127.0.0.1:8000


Documentação automática (Swagger):

http://127.0.0.1:8000/docs

📤 Exemplo de Requisição POST

Envie para a rota /predict:

{
  "last_60_days": [
    30.5, 31.0, 30.8, 30.7, 31.2, 31.5,
    31.3, 30.9, 30.4, 30.2,
    ... (total de 60 números) ...
  ]
}


Resposta esperada:

{
  "prediction": 32.487
}

📊 Avaliação do Modelo

O modelo utiliza as seguintes métricas:

MAE — Mean Absolute Error

RMSE — Root Mean Square Error

Essas métricas permitem avaliar o quão distante a previsão fica em relação ao valor real.

🐳 (Opcional) Executando com Docker

Build:

docker build -t stock-lstm-api .


Rodar o container:

docker run -p 8000:8000 stock-lstm-api


API disponível em:

http://localhost:8000

📌 Requisitos

Conteúdo do requirements.txt:

numpy
pandas
yfinance
scikit-learn
tensorflow
fastapi
uvicorn
joblib
matplotlib

📝 Observações Importantes

Caso os arquivos .h5 e .pkl não existam, a API mostrará um aviso e não funcionará até que o modelo seja treinado.

Para treinar outra ação, altere o símbolo no arquivo train_model.py:

symbol = 'PETR4.SA'
