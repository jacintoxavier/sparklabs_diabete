import pandas as pd
import numpy as np
import streamlit as st
# import joblib
import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model
from sklearn.preprocessing import LabelEncoder

#carregando o modelo treinado.
# var_model = "../2_Dados/model"
var_model = "../2_Dados/model_diabetes"
model = load_model(var_model)
print(type(model))
# model

#carregando o conjunto de dados.
var_dataset = "../2_Dados/Modelagem.parquet.gzip"
dataset = pd.read_parquet(var_dataset)
# dataset

# título
st.sidebar.title("Predição de Diabetes")

# subtítulo
st.subheader("Este é um Data App utilizado para exibir a solução de Machine Learning para o problema de Análise de predição de Diabetes.")

# imprime o conjunto de dados usado
# st.dataframe(dataset.head())

st.subheader("Defina os atributos da pessoa para predição de diabetes")
df1 = pd.DataFrame(np.array([[0,'Pressão Normal'],[1,'Pressão Alta'],[0,'Colesterol Normal'],[1,'Colesterol Alto'],[0,'Testou o colesterol nos últimos 5 anos'],[1,'Não testou colesterol nos últimos 5 anos'],[0,'Fumou menos de 100 cigarros em sua vida'],[1,'Fumou 100 cigarros ou mais em sua vida (5 pacotes = 100 cigarros)'],[0,'Não teve AVC'],[1,'Teve AVC'],[0,'Não teve ataque cardíaco'],[1,'Já teve ataque cardíaco'],[0,'Se exercitou nos últimos 30 dias'],[1,'Não exercitou nos últimos 30 dias'],[0,'Consome 1 ou mais fruta por dia'],[1,'Não consome nenhuma fruta diariamente'],[0,'Consome 1 ou mais vegetais por dia'],[1,'Não consome nenhum vegetal diariamente'],[0,'Não consome álcool (quantidade menores citadas para homem e mulher quando o valor é 1)'],[1,'Homem - consome mais de 14 drinks por semana e Mulher - consome mais de 7 drinks por semana'],[0,'Possui algum seguro de saúde'],[1,'Não possui seguro saúde'],[0,'Não precisou ir ao médico nos últimos 12 meses'],[1,'Precisou ir no médico nos últimos 12 meses, mas não foi pelo preço'],[1,'Excelente'],[2,'Muito Boa'],[3,'Satisfatório'],[4,'Ruim'],[5,'Horrível'],[0,'Não tem dificuldades de subir escada'],[1,'em dificuldades em subir escadas'],[0,'Feminino'],[1,'Masculino'],[1, '18 a 24 anos'],[2, '25 a 29 anos'],[3, '30 a 34 anos'],[4, '35 a 39 anos'],[5, '40 a 44 anos'],[6, '45 a 49 anos'],[7, '50 a 54 anos'],[8, '55 a 59 anos'],[9, '60 a 64 anos'],[10, '65 a 69 anos'],[11, '70 a 74 anos'],[12, '75 a 79 anos'],[13, '80 anos ou mais'],[1, 'Superior Completo'],[2, 'Superior Incompleto'],[3, 'Médio Completo '],[4, 'Médio Incompleto'],[5, 'Fundamental'],[6, 'Sem Educação'],[1, 'Maior que 75 mil'],[2, 'Maior que 50 mil e menor que 75 mil'],[3, 'Maior que 35 mil e menor que 50 mil'],[4, 'Maior que 25 mil e menor que 35 mil'],[5, 'Maior que 20 mil e menor que 25 mil'],[6, 'Maior que 15 mil e menor que 20 mil'],[7, 'Maior que 10 mil e menor que 15 mil'],[8, 'Menor que 10 mil']]),columns = ['cod','desc'])

# mapeando dados do usuário para cada atributo
cols = st.columns((1, 1,1))
# author = cols[0].text_input("Report author:")
pressao_alta = cols[0].selectbox(
     'Pressão alta ?',
     (df1['cod'][0] + ' - ' + df1['desc'][0] , df1['cod'][1] + ' - ' + df1['desc'][1])
     )

colesterol = cols[1].selectbox(
     'Colesterol ?',
     (df1['cod'][2] + ' - ' + df1['desc'][2] , df1['cod'][3] + ' - ' + df1['desc'][3])
     )

testou_colesterol = cols[2].selectbox(
     'Testou o colesterol ?',
     (df1['cod'][4] + ' - ' + df1['desc'][4] , df1['cod'][5] + ' - ' + df1['desc'][5])
     )

IMC = cols[0].number_input("IMC")

fumante = cols[1].selectbox(
     'Fumante ?',
     (df1['cod'][6] + ' - ' + df1['desc'][6] , df1['cod'][7] + ' - ' + df1['desc'][7])
     ) 

avc = cols[2].selectbox(
     'AVC ?',
     (df1['cod'][8] + ' - ' + df1['desc'][8] , df1['cod'][9] + ' - ' + df1['desc'][9])
     ) 

ataque_cardiaco = cols[0].selectbox(
     'Ataque cardíaco ?',
     (df1['cod'][10] + ' - ' + df1['desc'][10] , df1['cod'][11] + ' - ' + df1['desc'][11])
     )

ativ_fisica = cols[1].selectbox(
     'Atividade física ?',
     (df1['cod'][12] + ' - ' + df1['desc'][12] , df1['cod'][13] + ' - ' + df1['desc'][13])
     )          
                    
consome_frutas = cols[2].selectbox(
     'Consome Frutas ?',
     (df1['cod'][14] + ' - ' + df1['desc'][14] , df1['cod'][15] + ' - ' + df1['desc'][15])
     )

consome_vegetais = cols[0].selectbox(
     'Consome Vegetais ?',
     (df1['cod'][16] + ' - ' + df1['desc'][16] , df1['cod'][17] + ' - ' + df1['desc'][17])
     )

consumo_alcool = cols[1].selectbox(
     'Consome Alcool ?',
     (df1['cod'][18] + ' - ' + df1['desc'][18] , df1['cod'][19] + ' - ' + df1['desc'][19])
     )

seguro_saude = cols[2].selectbox(
     'Seguro Saude ?',
     (df1['cod'][20] + ' - ' + df1['desc'][20] , df1['cod'][21] + ' - ' + df1['desc'][21])
     )

medico_custo = cols[0].selectbox(
     'Custo Médico?',
     (df1['cod'][22] + ' - ' + df1['desc'][22] , df1['cod'][23] + ' - ' + df1['desc'][23])
     )

auto_aval_saude = cols[1].selectbox(
     'Auto avaliação Saúde',
     (df1['cod'][24] + ' - ' + df1['desc'][24] , df1['cod'][25] + ' - ' + df1['desc'][25] , df1['cod'][26] + ' - ' + df1['desc'][26] , df1['cod'][27] + ' - ' + df1['desc'][27] , df1['cod'][28] + ' - ' + df1['desc'][28])
     )

saude_mental = cols[2].number_input("Saude Mental", value=dataset["saude_mental"].mean())

saude_fisica = cols[0].number_input("Saude física", value=dataset["saude_fisica"].mean())

dif_escadas = cols[1].selectbox(
     'Dificuldade com  Escada ?',
     (df1['cod'][29] + ' - ' + df1['desc'][29] , df1['cod'][30] + ' - ' + df1['desc'][30])
     )  

sexo = cols[2].selectbox(
     'Sexo',
     (df1['cod'][31] + ' - ' + df1['desc'][31] , df1['cod'][32] + ' - ' + df1['desc'][32])
     )   

class_idade = cols[0].selectbox(
     'Idade',
     (df1['cod'][33] + ' - ' + df1['desc'][33], df1['cod'][34] + ' - ' + df1['desc'][34], df1['cod'][35] + ' - ' + df1['desc'][35], df1['cod'][36] + ' - ' + df1['desc'][36], df1['cod'][37] + ' - ' + df1['desc'][37], df1['cod'][38] + ' - ' + df1['desc'][38], df1['cod'][39] + ' - ' + df1['desc'][39], df1['cod'][40] + ' - ' + df1['desc'][40], df1['cod'][41] + ' - ' + df1['desc'][41], df1['cod'][42] + ' - ' + df1['desc'][42], df1['cod'][43] + ' - ' + df1['desc'][43], df1['cod'][44] + ' - ' + df1['desc'][44], df1['cod'][45] + ' - ' + df1['desc'][45])
     )   

class_educ = cols[1].selectbox(
     'Educação',
     (df1['cod'][46] + ' - ' + df1['desc'][46] , df1['cod'][47] + ' - ' + df1['desc'][47], df1['cod'][48] + ' - ' + df1['desc'][48], df1['cod'][49] + ' - ' + df1['desc'][49], df1['cod'][50] + ' - ' + df1['desc'][50], df1['cod'][51] + ' - ' + df1['desc'][51])
     )  

class_renda = cols[2].selectbox(
     'Renda',
     (df1['cod'][52] + ' - ' + df1['desc'][52], df1['cod'][53] + ' - ' + df1['desc'][53], df1['cod'][54] + ' - ' + df1['desc'][54], df1['cod'][55] + ' - ' + df1['desc'][55], df1['cod'][56] + ' - ' + df1['desc'][56], df1['cod'][57] + ' - ' + df1['desc'][57], df1['cod'][58] + ' - ' + df1['desc'][58], df1['cod'][59] + ' - ' + df1['desc'][59])
     )  


# inserindo um botão na tela
btn_predict = st.sidebar.button("Realizar Classificação")

st.sidebar.title("Resposta:")

# verifica se o botão foi acionado
if btn_predict:    
    # consome_frutas =  labelencoder.fit_transform(consome_frutas[0])
    print(consome_frutas[0])   
    # teste = np.array([[28.38,4.24,0.43,2.51,0.63,0.76,0.44,0.81]])  
    data_teste = np.array([[pressao_alta[0],colesterol[0],testou_colesterol[0],IMC,'20',avc[0],ataque_cardiaco[0],ativ_fisica[0],consome_frutas[0],consome_vegetais[0],consumo_alcool[0],seguro_saude[0],medico_custo[0],auto_aval_saude[0],saude_mental,saude_fisica,dif_escadas[0],sexo[0],class_idade[0],class_educ[0],class_renda[0]]])  
    data_teste = data_teste.astype(float)
    # data_teste2 = np.array([[0,0,0,0,20.0,0,0,0,0,0,0,0,0,0,1,3,4,0,0,0,1]])    
    # st.write(type(data_teste[0][0]))
    # st.write(data_teste2)
    # st.write(type(data_teste2[0][0])) 
    data_testeRes = pd.DataFrame()
    data_testeRes["pressao_alta"]           = [pressao_alta]
    data_testeRes["colesterol_alto"]        = [colesterol]
    data_testeRes["testou_colesterol_nova"] = [testou_colesterol]
    data_testeRes["IMC"]                    = [IMC]
    data_testeRes["fumante"]                = [fumante]
    data_testeRes["AVC"]                    = [avc]
    data_testeRes["ataque_cardiaco"]        = [ataque_cardiaco]
    data_testeRes["ativ_fisica_nova"]       = [ativ_fisica]
    data_testeRes["consome_frutas_nova"]    = [consome_frutas]
    data_testeRes["consome_vegetais_nova"]  = [consome_vegetais] 
    data_testeRes["consumo_alcool"]         = [consumo_alcool]
    data_testeRes["seguro_saude_nova"]      = [seguro_saude]
    data_testeRes["n_foi_med_custo"]        = [medico_custo]
    data_testeRes["auto_aval_saude"]        = [auto_aval_saude]
    data_testeRes["saude_mental"]           = [saude_mental]
    data_testeRes["saude_fisica"]           = [saude_fisica]
    data_testeRes["dif_escadas"]            = [dif_escadas]
    data_testeRes["sexo"]                   = [sexo]
    data_testeRes["class_idade"]            = [class_idade]
    data_testeRes["class_educ_nova"]        = [class_educ]
    data_testeRes["class_renda_nova"]       = [class_renda]   
   
    st.write(data_teste)
    st.write(data_testeRes)
    st.write(type(data_teste[0][0]))  
   
    result = model.predict(data_teste)    

    if(result[0] == 0):

        resposta = 'O modelo classificou como ausência de diabetes'
    

    if(result[0] == 1):

        resposta = 'O modelo classificou como presença de diabetes'


    
    st.sidebar.write(resposta) 
    
    # print(data_teste)
    