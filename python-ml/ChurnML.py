#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from ipywidgets import interact, interactive, interact_manual

import pandas as pd
import numpy as np
import lineup_widget
import sys
print(sys.argv)
import base64

import sqlalchemy
import mysql.connector
from sqlalchemy import types, create_engine

"""
Cria conexao ao banco de dados MySQL
"""
MYSQL_USER 		= 'root'
MYSQL_PASSWORD 	= 'setpassword'
MYSQL_HOST_IP 	= '127.0.0.1'
MYSQL_PORT		= '3306'
MYSQL_DATABASE	= 'Retencao'

engine = create_engine('mysql+mysqldb://'+MYSQL_USER+':'+MYSQL_PASSWORD+'@'+MYSQL_HOST_IP+':'+MYSQL_PORT+'/'+MYSQL_DATABASE, echo=False)
dbConnection    = engine.connect()
chunksize = 500


"""
Metodo para ler um csv e carregar os dados no MySQL
"""
def csvtosql(csv, tbl, separator, idx=1):
    for df in pd.read_csv(csv, chunksize=chunksize, sep=separator):
    	if idx == 1: 
    		exists = 'replace'
    	else:
    		exists = 'append'
    
    	df.to_sql(name=tbl, con=engine, if_exists=exists, index=False, chunksize=chunksize)
    	print(str(chunksize * idx)+" Processed");
    	idx = idx+1
        
        
"""
Le as fontes de dados em csv e carrega em MySQL
"""
"""
csvtosql('../data/ModeloCanceladosTreinamento.txt', 'treinamento','\t' )
csvtosql('../data/ModeloCanceladosClassificacao.txt', 'classificacao','\t')
csvtosql('../data/ClientesProdutos.csv', 'produtos',';')
csvtosql('../data/atendimento-fone.csv', 'atendimento_fone',';')
csvtosql('../data/atendimento-chamados.csv', 'atendimento_chamado',';')
"""


"""
Obtem os dados para treinamento
"""
#file=pd.read_csv(str(sys.argv[1]), sep='\t')
#file=pd.read_csv('../data/ModeloCanceladosTreinamento.txt', sep='\t')
file=pd.read_sql("select * from treinamento", dbConnection);
#pd.set_option('display.expand_frame_repr', False)
#print(file)


"""
Trata campos categoricos
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_previsores=LabelEncoder()

file['RedeCategoria']=labelencoder_previsores.fit_transform(file['Rede'])
file['PlataformaCategoria']=labelencoder_previsores.fit_transform(file['Plataforma'])
file['PlanoCategoria']=labelencoder_previsores.fit_transform(file['Plano'])
file['TituloSiteCategoria']=labelencoder_previsores.fit_transform(file['TituloSite'])
file['StatusFinanceiroCategoria']=labelencoder_previsores.fit_transform(file['StatusFinanceiro'])
file['PerfilPagamentoCategoria']=labelencoder_previsores.fit_transform(file['PerfilPagamento'])
file['ServidorCategoria']=labelencoder_previsores.fit_transform(file['Servidor'])
file['GeneroCategoria']=labelencoder_previsores.fit_transform(file['Genero'])
file['TipoPessoaCategoria']=labelencoder_previsores.fit_transform(file['TipoPessoa'])
file['EstadoCategoria']=labelencoder_previsores.fit_transform(file['Estado'])
file['NpsCategoria']=labelencoder_previsores.fit_transform(file['Nps'])
#file['TipoDominioCategoria']=labelencoder_previsores.fit_transform(file['TipoDominio'])

"""
Monta arquivos de treinamento e teste
"""
file_cancelados=file[file['StatusDominio']=="Cancelado"].copy()
file_ativos=file[file['StatusDominio']=="Ativo"][:file_cancelados['idCliente'].count()].copy()

file_treinamento=file_cancelados.append(file_ativos)

from sklearn.model_selection import train_test_split
carascteristicas=file_treinamento[['RedeCategoria',
                                   'PlataformaCategoria',
                                   'ServidorCategoria',
                                   'PlanoCategoria',
                                   'TituloSiteCategoria',
                                   'StatusFinanceiroCategoria',
                                   'PerfilPagamentoCategoria',
                                   'EspacoWeb',
                                   'EspacoImap',
                                   'EspacoBanco',
                                   'TrafegoFtp',
                                   'MediaMemoriaDiaria',
                                   'MediaCpuDiariaTotal',
                                   'Visitas',
                                   'CaixasEmail',
                                   'EnviosEmail',
                                   'DiasAtivo',
                                   'ServidorTotalQuedas',
                                   'ServidorTempoQuedas',
                                   'Contatos',
                                   'Idade',
                                   'TipoPessoaCategoria',
                                   'EstadoCategoria',
                                   'NpsCategoria']].copy()

classificacao=file_treinamento['StatusDominio'].copy()

X_train, X_test, y_train, y_test = train_test_split(carascteristicas, classificacao, test_size=0.15, random_state=0)

"""
Cria árvore de decisão para classificar domínios propensos ao cancelamento
"""
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=1000,random_state=42)
clf=clf.fit(X_train,y_train)
clf.score(X_test, y_test)

from sklearn.metrics import precision_recall_fscore_support as score
precision, recall, fscore, support = score(y_test, clf.predict(X_test))
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

feature_list=['RedeCategoria',
                                   'PlataformaCategoria',
                                   'ServidorCategoria',
                                   'PlanoCategoria',
                                   'TituloSiteCategoria',
                                   'StatusFinanceiroCategoria',
                                   'PerfilPagamentoCategoria',
                                   'EspacoWeb',
                                   'EspacoImap',
                                   'EspacoBanco',
                                   'TrafegoFtp',
                                   'MediaMemoriaDiaria',
                                   'MediaCpuDiariaTotal',
                                   'Visitas',
                                   'CaixasEmail',
                                   'EnviosEmail',
                                   'DiasAtivo',
                                   'ServidorTotalQuedas',
                                   'ServidorTempoQuedas',
                                   'Contatos',
                                   'Idade',
                                   'TipoPessoaCategoria',
                                   'EstadoCategoria',
                                   'NpsCategoria']

"""
Verifica a importancia de cada campo
"""
# Get numerical feature importances
importances = list(clf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];



"""
Consulta domínios para classificar
"""
fileClassificacao=pd.read_sql("select * from classificacao", dbConnection);
#fileClassificacao=pd.read_csv('../data/ModeloCanceladosClassificacao.txt', sep='\t')
#fileClassificacao=pd.read_csv(str(sys.argv[2]), sep='\t')


#Trata campos categoricos
fileClassificacao['RedeCategoria']=labelencoder_previsores.fit_transform(fileClassificacao['Rede'])
fileClassificacao['PlataformaCategoria']=labelencoder_previsores.fit_transform(fileClassificacao['Plataforma'])
fileClassificacao['PlanoCategoria']=labelencoder_previsores.fit_transform(fileClassificacao['Plano'])
fileClassificacao['TituloSiteCategoria']=labelencoder_previsores.fit_transform(fileClassificacao['TituloSite'])
fileClassificacao['StatusFinanceiroCategoria']=labelencoder_previsores.fit_transform(fileClassificacao['StatusFinanceiro'])
fileClassificacao['PerfilPagamentoCategoria']=labelencoder_previsores.fit_transform(fileClassificacao['PerfilPagamento'])
fileClassificacao['ServidorCategoria']=labelencoder_previsores.fit_transform(fileClassificacao['Servidor'])
fileClassificacao['GeneroCategoria']=labelencoder_previsores.fit_transform(fileClassificacao['Genero'])
fileClassificacao['TipoPessoaCategoria']=labelencoder_previsores.fit_transform(fileClassificacao['TipoPessoa'])
fileClassificacao['EstadoCategoria']=labelencoder_previsores.fit_transform(fileClassificacao['Estado'])
fileClassificacao['NpsCategoria']=labelencoder_previsores.fit_transform(fileClassificacao['Nps'])



"""
apply(self, X)

Apply trees in the forest to X, return leaf indices.

decision_path(self, X)

Return the decision path in the forest.

fit(self, X, y[, sample_weight])

Build a forest of trees from the training set (X, y).

get_params(self[, deep])

Get parameters for this estimator.

predict(self, X)

Predict class for X.

predict_log_proba(self, X)

Predict class log-probabilities for X.

predict_proba(self, X)

Predict class probabilities for X.

score(self, X, y[, sample_weight])

Return the mean accuracy on the given test data and labels.

set_params(self, \*\*params)

Set the parameters of this estimator.
"""

fileClassificacao['Resultado']=clf.predict(fileClassificacao[['RedeCategoria',
                                   'PlataformaCategoria',
                                   'ServidorCategoria',
                                   'PlanoCategoria',
                                   'TituloSiteCategoria',
                                   'StatusFinanceiroCategoria',
                                   'PerfilPagamentoCategoria',
                                   'EspacoWeb',
                                   'EspacoImap',
                                   'EspacoBanco',
                                   'TrafegoFtp',
                                   'MediaMemoriaDiaria',
                                   'MediaCpuDiariaTotal',
                                   'Visitas',
                                   'CaixasEmail',
                                   'EnviosEmail',
                                   'DiasAtivo',
                                   'ServidorTotalQuedas',
                                   'ServidorTempoQuedas',
                                   'Contatos',
                                   'Idade',
                                   'TipoPessoaCategoria',
                                   'EstadoCategoria',
                                   'NpsCategoria']])
    

#retorna probabilidade para o resultado alvo
fileClassificacao['Probabilidade']=clf.predict_proba(fileClassificacao[['RedeCategoria',
                                   'PlataformaCategoria',
                                   'ServidorCategoria',
                                   'PlanoCategoria',
                                   'TituloSiteCategoria',
                                   'StatusFinanceiroCategoria',
                                   'PerfilPagamentoCategoria',
                                   'EspacoWeb',
                                   'EspacoImap',
                                   'EspacoBanco',
                                   'TrafegoFtp',
                                   'MediaMemoriaDiaria',
                                   'MediaCpuDiariaTotal',
                                   'Visitas',
                                   'CaixasEmail',
                                   'EnviosEmail',
                                   'DiasAtivo',
                                   'ServidorTotalQuedas',
                                   'ServidorTempoQuedas',
                                   'Contatos',
                                   'Idade',
                                   'TipoPessoaCategoria',
                                   'EstadoCategoria',
                                   'NpsCategoria']])[:,1]
    


"""
# ver a arvore gerada e o impacto das variaveis
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = clf.estimators_[5]
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = clf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')
"""

"""
Verifica a importancia de cada campo
"""
# Get numerical feature importances
importances = list(clf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


"""
Trata os dados com binning para intervalos menores (0-100) visando facilitar a futura visualizacao
"""
fileDashGeral=fileClassificacao
#fileDashGeral=fileClassificacao[fileClassificacao['resultado']=="Cancelado"].copy()


def binning(col, cut_points, labels=None):
  #Define min and max values:
  minval = col.min()
  maxval = col.max()

  #cria lista adicionando min e max a cut_points
  break_points = [minval] + cut_points + [maxval]

  #se nenhum rótulo for definido, usa rótulos default 0 ... (n-1)
  if not labels:
    labels = range(len(cut_points)+1)

  #Binning usando a função cut
  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True, duplicates='drop')
  return colBin

#usa funcao de percentil para identificar os pontos de corte 
def binningValue(field, fieldType):
    p10 = np.percentile(field[lambda x: x!=0], 10)
    p20 = np.percentile(field[lambda x: x!=0], 20)
    p30 = np.percentile(field[lambda x: x!=0], 30)
    p40 = np.percentile(field[lambda x: x!=0], 40)
    p50 = np.percentile(field[lambda x: x!=0], 50)
    p60 = np.percentile(field[lambda x: x!=0], 60)
    p70 = np.percentile(field[lambda x: x!=0], 70)
    p80 = np.percentile(field[lambda x: x!=0], 80)
    p90 = np.percentile(field[lambda x: x!=0], 90)

    print(p10, p20,p30,p40,p50,p60,p70,p80,p90)
    
    if fieldType == "completo":
        cut_points = [p10, p20,p30,p40,p50,p60,p70,p80,p90]
        labels = [10,20,30,40,50,60,70,80,90, 100]
    if fieldType == "menor":
        cut_points = [p30,p60,p90]
        labels = [0,33,66,100]

    return binning(field, cut_points, labels)
    
    
  
fileDashGeral['DiasAtivo'] = binningValue(fileDashGeral['DiasAtivo'], "completo")
fileDashGeral['EnviosEmail'] = binningValue(fileDashGeral['EnviosEmail'], "menor")
fileDashGeral['Visitas'] = binningValue(fileDashGeral['Visitas'], "menor")
fileDashGeral['EspacoWeb'] = binningValue(fileDashGeral['EspacoWeb'], "completo")
fileDashGeral['EspacoImap'] = binningValue(fileDashGeral['EspacoImap'], "completo")
fileDashGeral['EspacoBanco'] = binningValue(fileDashGeral['EspacoBanco'], "completo")
fileDashGeral['TrafegoFtp'] = binningValue(fileDashGeral['TrafegoFtp'], "completo")
fileDashGeral['MediaMemoriaDiaria'] = binningValue(fileDashGeral['MediaMemoriaDiaria'], "completo")
fileDashGeral['MediaCpuDiariaTotal'] = binningValue(fileDashGeral['MediaCpuDiariaTotal'], "completo")
fileDashGeral['ServidorTotalQuedas'] = binningValue(fileDashGeral['ServidorTotalQuedas'], "menor")
fileDashGeral['ServidorTempoQuedas'] = binningValue(fileDashGeral['ServidorTempoQuedas'], "menor")
fileDashGeral['Contatos'] = binningValue(fileDashGeral['Contatos'], "menor")
fileDashGeral['Probabilidade'] = fileDashGeral['Probabilidade'] * 100


MODE_ENCRYPT = 1
MODE_DECRYPT = 0

def anonymize(data, key, mode):
    alphabet = 'abcdefghijklmnopqrstuvwyzàáãâéêóôõíúçABCDEFGHIJKLMNOPQRSTUVWYZÀÁÃÂÉÊÓÕÍÚÇ'
    new_data = ''
    for c in data:
        index = alphabet.find(c)
        if index == -1:
            new_data = c
        else:
            new_index = index + key if mode == MODE_ENCRYPT else index - key
            new_index = new_index % len(alphabet)
            new_data = alphabet[new_index:new_index+1]
    return new_data


#fileDashGeral['Dominio'] = anonymize(fileDashGeral['Dominio'], 5, MODE_ENCRYPT)
#print    (fileDashGeral['Dominio'])
print (anonymize(fileDashGeral['Dominio'], 5, MODE_ENCRYPT))


"""
#normaliza valores entre 0 e 100
for data in fileDashGeral:
   # print(fileDashGeral[data].dtype)
    var = (fileDashGeral[data])
    if (fileDashGeral[data].dtype != object):
    #    print('STR NAO')
        fileDashGeral[data] = (fileDashGeral[data]/max(fileDashGeral[data]))*100
    else:
     # print('STR sim')  
      fileDashGeral[data] = fileDashGeral[data]  
        
"""

def sqlcol(dfparam):    

    dtypedict = {}
    for i,j in zip(dfparam.columns, dfparam.dtypes):
        
        print(i, str(j))
        
        if "object" in str(j):
            dtypedict.update({i: sqlalchemy.types.NVARCHAR(length=255)})
            #dtypedict.update({i: sqlalchemy.types.Enum()})

        if "datetime" in str(j):
            dtypedict.update({i: sqlalchemy.types.DateTime()})

        if "float" in str(j):
            dtypedict.update({i: sqlalchemy.types.Float(precision=3, asdecimal=True)})

        if "int" in str(j):
            dtypedict.update({i: sqlalchemy.types.INT()})
           
        if "category" in str(j):
            #dtypedict.update({i: sqlalchemy.types.INT()})
            dtypedict.update({i: sqlalchemy.types.Numeric()})

    return dtypedict

"""
Grava os resultados em banco de dados
"""

outputdict = sqlcol(fileDashGeral)
print (outputdict)
fileDashGeral.to_sql(name="resultado", 
                                       con=engine, 
                                       if_exists="replace", 
                                       index=False,
                                       dtype=outputdict)



"""
fileDashGeral[['Dominio',
                   'prob',
                   'DiasAtivo',
                   'EnviosEmail',
                   'Visitas',
                   'EspacoWeb',
                   'EspacoImap',
                   'EspacoBanco',
                   'TrafegoFtp',
                   'MediaMemoriaDiaria',
                   'MediaCpuDiariaTotal',
                   'ServidorTotalQuedas',
                   'ServidorTempoQuedas', 
                   'Contatos']].to_sql(name="resultado", 
                                       con=engine, 
                                       if_exists="replace", 
                                       index=False,
                                       dtype=outputdict)
#to_csv('../data/resultado1.txt', index=False)                

                                       
                                       
                   

fileClassificacao[['idDominio',
                  'Dominio',
                  'RedeCategoria',
                  'PlataformaCategoria',
                  'ServidorCategoria',
                  'PlanoCategoria',
                  'TituloSiteCategoria',
                  'StatusFinanceiroCategoria',
                  'PerfilPagamentoCategoria',
                  'EspacoWeb',
                  'EspacoImap',
                  'EspacoBanco',
                  'TrafegoFtp',
                  'MediaMemoriaDiaria',
                  'MediaCpuDiariaTotal',
                  'Visitas',
                  'CaixasEmail',
                  'EnviosEmail',
                  'DiasAtivo',
                  'ServidorTotalQuedas',
                  'ServidorTempoQuedas',
                  'Contatos',
                  'Idade',
                  'TipoPessoaCategoria',
                  'EstadoCategoria',
                  'NpsCategoria',                                  
                  'resultado',
                  'prob']].to_sql(name="resultado2", con=engine, if_exists="replace", index=False)
#to_csv('../data/resultado2.txt', index=False)
"""

"""

df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))

w = lineup_widget.LineUpWidget(df)
w.on_selection_changed(lambda selection: print(selection))
w


def selection_changed(selection):
    return df.iloc[selection]

interact(selection_changed, selection=lineup_widget.LineUpWidget(df));


w = lineup_widget.LineUpWidget(fileDashGeral)
w.on_selection_changed(lambda selection: print(selection))
w
"""


dbConnection.close()

