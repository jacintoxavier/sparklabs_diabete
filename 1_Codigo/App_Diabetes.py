import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model

#carregando o modelo treinado.
var_model = "../2_Dados/model"
model = load_model(var_model)
# model

#carregando o conjunto de dados.
var_dataset = "../2_Dados/Modelagem.parquet.gzip"
dataset = pd.read_parquet(var_dataset)
# dataset

# título
st.title("Análise de predição de Diabetes")

# subtítulo
st.markdown("Este é um Data App utilizado para exibir a solução de Machine Learning para o problema de Análise de predição de Diabetes.")

# imprime o conjunto de dados usado
st.dataframe(dataset.head())

st.sidebar.subheader("Defina os atributos da pessoa para predição de diabetes")

# mapeando dados do usuário para cada atributo
IMC = st.sidebar.number_input("IMC", value=dataset["IMC"].mean())
saude_fisica = st.sidebar.number_input("saude_fisica", value=dataset["saude_fisica"].mean())
pressao_alta = st.sidebar.number_input("pressao_alta", value=dataset["pressao_alta"].mean())
auto_aval_saude = st.sidebar.number_input("auto_aval_saude", value=dataset["auto_aval_saude"].mean())
consome_frutas = st.sidebar.number_input("consome_frutas", value=dataset["consome_frutas"].mean())
ativ_fisica = st.sidebar.number_input("ativ_fisica", value=dataset["ativ_fisica"].mean())
fumante = st.sidebar.number_input("fumante", value=dataset["fumante"].mean())
consome_vegetais = st.sidebar.number_input("consome_vegetais", value=dataset["consome_vegetais"].mean())

# inserindo um botão na tela
btn_predict = st.sidebar.button("Realizar Classificação")

# verifica se o botão foi acionado
if btn_predict:
    data_teste = pd.DataFrame()
    data_teste["IMC"] = [IMC]
    data_teste["saude_fisica"] = [saude_fisica]    
    data_teste["pressao_alta"] = [pressao_alta]
    data_teste["auto_aval_saude"] = [auto_aval_saude]
    data_teste["consome_frutas"] = [consome_frutas]
    data_teste["ativ_fisica"] = [ativ_fisica]
    data_teste["fumante"] = [fumante]
    data_teste["consome_vegetais"] = [consome_vegetais] 

    #imprime os dados de teste    
    print(data_teste)

    # model2 = joblib.load('../2_Dados/model.pkl')
    # import numpy as np
    # teste = np.array(data_teste)

    # print('Resultado ',model2.predict(teste))

    #realiza a predição
    # save_config(data_teste)
    # df = data_teste.reset_index()
    # save_config('test')
    # load_model('test')
    # result = predict_model(model, data=data_teste)
    result = model.predict(data_teste)

    # print('result ',result[0])
    if(result[0] == 0):

        resposta = 'O modelo classificou como ausência de diabetes'
    

    if(result[0] == 1):

        resposta = 'O modelo classificou como presença de diabetes'


    st.title("Resposta:")
    st.write(resposta) 