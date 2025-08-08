

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 09:50:00 2025
@author: fabio.melchior

SCRIPT DE CLASSIFICAÇÃO MULTIRRÓTULO (v2)
-----------------------------------------------
Este script foi ajustado para:
1. Sempre exibir os nomes completos dos rótulos nos relatórios e gráficos.
2. Utilizar a configuração de parâmetros manuais pré-GridSearchCV como padrão.
"""

import pandas as pd
# import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, precision_score, recall_score, jaccard_score, classification_report

# Bibliotecas para visualização
#import matplotlib.pyplot as plt
#import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import sys

#!pip install joblib
import joblib


#Configurações Iniciais e Download de Recursos NLTK 
try: stopwords.words('portuguese')
except LookupError: nltk.download('stopwords')
try: nltk.word_tokenize("exemplo")
except LookupError: nltk.download('punkt')
try: RSLPStemmer()
except LookupError: nltk.download('rslp')

# ===================================
#1. CONFIGURAÇÃO DO EXPERIMENTO 
# ===================================
print("--- INICIANDO EXPERIMENTO DE CLASSIFICAÇÃO MULTIRRÓTULO ")
print("="*60)

#Parâmetros do Usuário 
NOME_ARQUIVO_XLSX = "planilha.xlsx"
NOME_DA_ABA = "Aba 1"
COLUNA_TEXTO = 'Content'
SUPORTE_MINIMO_PARA_MODELAGEM = 50

# Mantenha como False para usar os parâmetros manuais restaurados.
EXECUTAR_GRID_SEARCH = False 

# ===================================
#2. CARREGAMENTO E PREPARAÇÃO DOS DADOS 
# ===================================
print("\n--- 2. CARREGANDO E PREPARANDO OS DADOS ")
try:
    df = pd.read_excel(NOME_ARQUIVO_XLSX, sheet_name=NOME_DA_ABA)
    df.columns = df.columns.astype(str)
    print(f"-> Arquivo '{NOME_ARQUIVO_XLSX}' (aba: '{NOME_DA_ABA}') carregado com sucesso.")
except Exception as e:
    sys.exit(f"ERRO CRÍTICO ao ler o arquivo Excel: {e}\nCertifique-se de que o arquivo está no local correto e que a biblioteca 'openpyxl' está instalada.")

#Identificação e Filtragem Dinâmica dos Rótulos 
colunas_detectadas = df.columns.tolist()
COLUNAS_ROTULOS_ORIGINAIS = [col for col in colunas_detectadas if col not in ['Fonte', 'Content']]
if COLUNA_TEXTO not in df.columns: sys.exit(f"ERRO CRÍTICO: A coluna de texto '{COLUNA_TEXTO}' não foi encontrada no arquivo.")
for col in COLUNAS_ROTULOS_ORIGINAIS:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

contagens_rotulos = df[COLUNAS_ROTULOS_ORIGINAIS].sum()
COLUNAS_ROTULOS = contagens_rotulos[contagens_rotulos >= SUPORTE_MINIMO_PARA_MODELAGEM].index.tolist()

if not COLUNAS_ROTULOS: sys.exit(f"ERRO CRÍTICO: Nenhum rótulo atendeu ao suporte mínimo de {SUPORTE_MINIMO_PARA_MODELAGEM}.")

print(f"\n-> Número de rótulos a serem usados na modelagem: {len(COLUNAS_ROTULOS)}")
df.dropna(subset=[COLUNA_TEXTO], inplace=True)
df = df.reset_index(drop=True)
print(f"-> Total de decisões a serem processadas: {len(df)}")

# ===================================
#3. PRÉ-PROCESSAMENTO DO TEXTO 
# ===================================
print("\n--- 3. PRÉ-PROCESSANDO OS TEXTOS ")
stop_words_pt_nltk = set(stopwords.words('portuguese'))
stemmer_pt = RSLPStemmer()
def preprocess_text_rslp(text):
    if not isinstance(text, str): return ""
    text = text.lower(); text = re.sub(r'\d+', ' ', text); text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = nltk.word_tokenize(text, language='portuguese')
    return " ".join([stemmer_pt.stem(token) for token in tokens if token not in stop_words_pt_nltk and len(token) > 2])

df['texto_processado'] = df[COLUNA_TEXTO].astype(str).apply(preprocess_text_rslp)
print("-> Pré-processamento concluído.")

#Preparação final e divisão dos dados 
X = df['texto_processado']
Y = df[COLUNAS_ROTULOS]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
print(f"-> Dados divididos em {len(X_train)} amostras para treino e {len(X_test)} para teste.")

# ===================================
#4. MODELAGEM E TREINAMENTO 
# ===================================
print("\n--- 4. CONFIGURANDO O PIPELINE DE MODELAGEM ")
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', OneVsRestClassifier(estimator=RandomForestClassifier(random_state=42, n_jobs=-1)))
])

#**Parâmetros restaurados para a versão pré-GridSearchCV** 
manual_params = {
    'tfidf__ngram_range':(1,2), 
    'tfidf__min_df': 3, 
    'tfidf__max_df': 0.8,
    'tfidf__max_features': 5000, 
    'clf__estimator__n_estimators': 150,
    'clf__estimator__max_depth': None, 
    'clf__estimator__min_samples_leaf': 2,
    'clf__estimator__min_samples_split': 5, 
    'clf__estimator__class_weight': 'balanced_subsample'
}

if EXECUTAR_GRID_SEARCH:
    print("\n--- 4a. OTIMIZANDO HIPERPARÂMETROS COM GridSearchCV ")
    param_grid = { 'tfidf__min_df': [3, 5], 'clf__estimator__n_estimators': [150, 200] }
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='f1_macro', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, Y_train)
    multi_label_model = grid_search.best_estimator_
else:
    print("\n--- 4a. TREINANDO MODELO COM PARÂMETROS MANUAIS ")
    print(f"-> Parâmetros utilizados: {manual_params}")
    model_pipeline.set_params(**manual_params)
    multi_label_model = model_pipeline
    multi_label_model.fit(X_train, Y_train)
    print("-> Treinamento concluído.")

# ===================================
#5. ANÁLISE DO VETORIZADOR TF-IDF 
# ===================================
# (A Seção 5, com os relatórios de TF-IDF, permanece a mesma)
print("\n--- 5. GERANDO RELATÓRIO DO TF-IDF ")
try:
    tfidf_vectorizer = multi_label_model.named_steps['tfidf']
    feature_names = tfidf_vectorizer.get_feature_names_out()
    # ... (código restante da geração de relatórios TF-IDF)
except Exception as e_tfidf:
    print(f"Não foi possível gerar o relatório TF-IDF. Erro: {e_tfidf}")


# ===================================
#6. AVALIAÇÃO DO MODELO 
# ===================================
print("\n--- 6. AVALIANDO O DESEMPENHO NO CONJUNTO DE TESTE ")
Y_pred = multi_label_model.predict(X_test)

#Métricas Globais 
print("\n--- Métricas Agregadas ")
subset_accuracy = accuracy_score(Y_test, Y_pred)
hamming = hamming_loss(Y_test, Y_pred)
jaccard_sample_avg = jaccard_score(Y_test, Y_pred, average='samples', zero_division=0)
precision_sample_avg = precision_score(Y_test, Y_pred, average='samples', zero_division=0)
recall_sample_avg = recall_score(Y_test, Y_pred, average='samples', zero_division=0)
f1_sample_avg = f1_score(Y_test, Y_pred, average='samples', zero_division=0)

print(f"-> Subset Accuracy: {subset_accuracy:.4f}")
print(f"-> Hamming Loss: {hamming:.4f}")
print(f"-> Jaccard Score: {jaccard_sample_avg:.4f}")
print(f"-> Precision (Samples): {precision_sample_avg:.4f}")
print(f"-> Recall (Samples): {recall_sample_avg:.4f}")
print(f"-> F1-Score (Samples): {f1_sample_avg:.4f}")

#Relatório Detalhado por Rótulo 
print("\n--- Relatório de Classificação (Desempenho por Rótulo Individual) ")
report_dict = None
try:
    #**Forçando o uso dos nomes dos rótulos** 
    Y_pred_df = pd.DataFrame(Y_pred, columns=COLUNAS_ROTULOS, index=Y_test.index)
    report_dict = classification_report(Y_test, Y_pred_df, target_names=COLUNAS_ROTULOS, zero_division=0, output_dict=True)
    report_str = classification_report(Y_test, Y_pred_df, target_names=COLUNAS_ROTULOS, zero_division=0)
    print(report_str)
except Exception as e_report:
    print(f"Erro inesperado ao gerar classification_report: {e_report}")

# ===================================
#7. VISUALIZAÇÃO DOS DADOS E RESULTADOS 
# ===================================
print("\n--- 7. GERANDO VISUALIZAÇÕES FINAIS ")

#Gráfico de F1-Score por Rótulo (Interativo) 
if report_dict:
    df_report = pd.DataFrame(report_dict).transpose()
    df_f1_plotly = df_report[['precision', 'recall', 'f1-score', 'support']].iloc[:len(COLUNAS_ROTULOS)].reset_index()
    df_f1_plotly.rename(columns={'index': 'Rótulo'}, inplace=True)
    fig_f1_plotly = px.bar(
        df_f1_plotly.sort_values(by='f1-score', ascending=True), x='f1-score', y='Rótulo',
        orientation='h', title='Desempenho (F1-Score) Interativo por Rótulo',
        labels={'f1-score': 'F1-Score', 'Rótulo': 'Rótulos'}, text_auto='.2f',
        hover_data=['precision', 'recall', 'support']
    )
    fig_f1_plotly.update_xaxes(range=[0, 1])
    fig_f1_plotly.update_layout(yaxis={'categoryorder':'total ascending'})
    fig_f1_plotly.write_html("grafico_f1_interativo.html")
    print("-> Gráfico 'grafico_f1_interativo.html' salvo.")
else:
    print("-> Pulo na geração do gráfico de F1-score interativo.")

#Painel de Indicadores para Métricas Agregadas (Interativo) 
fig_indicators = go.Figure()
positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
metrics_to_plot = {
    "Subset Accuracy": subset_accuracy, "Hamming Loss": hamming,
    "Jaccard Score": jaccard_sample_avg, "Precision": precision_sample_avg,
    "Recall": recall_sample_avg
}
for i, (title, value) in enumerate(metrics_to_plot.items()):
    fig_indicators.add_trace(go.Indicator(
        mode="number", value=value,
        title={"text": f"<b>{title}</b><br><span style='font-size:0.8em;color:gray'>{'Menor é Melhor' if 'Loss' in title else 'Maior é Melhor'}</span>"},
        number={'valueformat': ".4f"}, domain={'row': positions[i][0], 'column': positions[i][1]}
    ))
fig_indicators.update_layout(
    grid={'rows': 2, 'columns': 3, 'pattern': "independent"},
    title_text="Painel de Métricas Agregadas de Desempenho", title_x=0.5,
    margin=dict(l=20, r=20, t=60, b=20)
)
fig_indicators.write_html("grafico_metricas_agregadas.html")
print("-> Gráfico 'grafico_metricas_agregadas.html' salvo.")

##################################################
###
###SALVANDO O MODELO E OS RÓTULOS PARA PRODUÇÃO 
###  Para utilizar no script aplicar_modelo_em_novos_dados.py
##################################################


print("\n--- SALVANDO ARQUIVOS PARA PRODUÇÃO ")
NOME_ARQUIVO_MODELO = 'modelo_classificador_final 4000.joblib'
NOME_ARQUIVO_ROTULOS = 'lista_de_rotulos_final 4000.joblib'

# Salva o pipeline completo (vetorizador + classificador)
joblib.dump(model_pipeline, NOME_ARQUIVO_MODELO)
print(f"-> Modelo salvo em: {NOME_ARQUIVO_MODELO}")

# Salva a lista de nomes de rótulos que o modelo aprendeu a prever
joblib.dump(COLUNAS_ROTULOS, NOME_ARQUIVO_ROTULOS)
print(f"-> Lista de rótulos salva em: {NOME_ARQUIVO_ROTULOS}")

print("\n--- PROCESSO DE TREINAMENTO E SALVAMENTO CONCLUÍDO ")





print("\n--- FIM DA EXECUÇÃO ")