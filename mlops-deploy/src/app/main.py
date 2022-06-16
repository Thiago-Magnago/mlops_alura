from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import pickle
import os

#Carrega o arquivo do modelo com permissão de leitura
modelo = pickle.load(open('../../models/modelo.sav', 'rb'))

#Colunas de entrada de dados
colunas = ['tamanho', 'ano', 'garagem']

#Criar o app/instanciar a classe Flask para o app com o nome 'meu_app'
app = Flask(__name__)
#Configuração para autenticação básica, via variáveis de ambiente
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)

#Definir as rotas da API: Rotas que os usuários/desenvolvedores acessarão
#Nossa rota base (home), acessa com a '/'
@app.route('/')

#Definindo a função a ser executada quando alguem chegar nessa rota
def home():
    return "Minha primeira API."

#Criando uma nova rota/endpoint: URL de acesso à nossa aplicação
#É o caminho desse endpoint na nossa API. Já receberá uma frase na URL
@app.route('/sentimento/<frase_english>')
@basic_auth.required #Informando que este endpoint precisa de autenticação
#Definindo a função que será executada
def sentimento(frase_english):
    tb_en = TextBlob(frase_english)
    polaridade = tb_en.sentiment.polarity
    return "polaridade: {}".format(polaridade)

#Criando outro endpoint para predição do preço da casa, que receberá um payload (conjunto de dados json)
@app.route('/cotacao/', methods=['POST']) #o input do usuário tem que ser do tipo inteiro
@basic_auth.required #Informando que este endpoint precisa de autenticação
def cotacao():
    dados = request.get_json() #Método que traz o json enviado pelo usuário
    dados_input = [dados[col] for col in colunas] #Faz o 'parser do input do usuário, garantindo a ordem de entrada conforme variável 'colunas'
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0]) #Devolvendo resultado em json: 1º e único elemento do array da predição do modelo


#Para o script rodar essa aplicação
#com debug=True, alterações aqui restartam a execução no terminal automaticamente
#com host=0.0.0.0, trata o fato de fazermos o deploy em vários ambientes: Docker, App Engine, local.
app.run(debug=True, host='0.0.0.0')