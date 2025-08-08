#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 18:57:54 2025

@author: fabio.melchior
"""



import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import sys
import joblib # Importante para carregar o modelo

# ===================================
# CONFIGURAÇÃO DA CLASSIFICAÇÃO 
# ===================================
print("--- INICIANDO CLASSIFICAÇÃO DE NOVAS DECISÕES ---")
print("="*60)

# --- Arquivos a serem utilizados ---
NOME_ARQUIVO_MODELO = 'modelo_classificador_final 4000.joblib'
NOME_ARQUIVO_ROTULOS = 'lista_de_rotulos_final 4000.joblib'
# !! IMPORTANTE: Aqui devemos informar nome do nosso arquivo Excel com os novos dados (nao rotulados - ou seja, que serao rotulados pelo modelo)
NOME_ARQUIVO_ENTRADA = 'Base de Dados Emenda sem rótulo.xlsx'
# Nome do arquivo que será gerado com as classificações feitas pelo modelo
NOME_ARQUIVO_SAIDA = 'Emendas_classificadas - 4000 - Emendas.xlsx'
COLUNA_TEXTO = 'Content' # Nome da coluna com o texto a ser classificado

# ===================================
# CARREGAMENTO DO MODELO E DOS NOVOS DADOS 
# ===================================
print("\n--- CARREGANDO MODELO TREINADO E NOVOS DADOS ---")
try:
    # Carrega o modelo e a lista de rótulos salvos na Etapa 1
    loaded_model = joblib.load(NOME_ARQUIVO_MODELO)
    loaded_label_names = joblib.load(NOME_ARQUIVO_ROTULOS)
    print("-> Modelo e lista de rótulos carregados com sucesso.")
except FileNotFoundError:
    sys.exit(f"Arquivos de modelo não encontrados. Certifique-se de que '{NOME_ARQUIVO_MODELO}' e '{NOME_ARQUIVO_ROTULOS}' estão na mesma pasta.")

try:
    # Carrega o novo arquivo Excel que você quer classificar
    df_novos_dados = pd.read_excel(NOME_ARQUIVO_ENTRADA)
    print(f"-> Arquivo '{NOME_ARQUIVO_ENTRADA}' com {len(df_novos_dados)} novas decisões carregado.")
except FileNotFoundError:
    sys.exit(f"Arquivo de entrada '{NOME_ARQUIVO_ENTRADA}' não encontrado.")
except Exception as e:
    sys.exit(f"ERRO CRÍTICO ao ler o arquivo de entrada: {e}")

if COLUNA_TEXTO not in df_novos_dados.columns:
    sys.exit(f"A coluna de texto '{COLUNA_TEXTO}' não foi encontrada no arquivo de entrada.")

# ===================================
# PRÉ-PROCESSAMENTO DOS NOVOS TEXTOS 
# ===================================
print("\n--- APLICANDO PRÉ-PROCESSAMENTO ---")
# É ESSENCIAL usar EXATAMENTE a mesma função de pré-processamento do treinamento
stop_words_pt_nltk = set(stopwords.words('portuguese'))
stemmer_pt = RSLPStemmer()
def preprocess_text_rslp(text):
    if not isinstance(text, str): return ""
    text = text.lower(); text = re.sub(r'\d+', ' ', text); text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = nltk.word_tokenize(text, language='portuguese')
    return " ".join([stemmer_pt.stem(token) for token in tokens if token not in stop_words_pt_nltk and len(token) > 2])

df_novos_dados['texto_processado'] = df_novos_dados[COLUNA_TEXTO].astype(str).apply(preprocess_text_rslp)
print("-> Pré-processamento dos novos textos concluído.")

# ===================================
#CLASSIFICAÇÃO (PREDIÇÃO)
# ===================================
print("\n--- REALIZANDO A CLASSIFICAÇÃO ---")
# O modelo usa a coluna de texto processado para fazer as previsões
predictions = loaded_model.predict(df_novos_dados['texto_processado'])
print("-> Classificação concluída.")

# ===================================
###FORMATAÇÃO E EXPORTAÇÃO DO RESULTADO 
# ===================================
print("\n--- PREPARANDO E SALVANDO O ARQUIVO DE SAÍDA ---")
# Converte o array de predições (0s e 1s) em um DataFrame com os nomes corretos dos rótulos
df_predictions = pd.DataFrame(predictions, columns=loaded_label_names)

# Junta o DataFrame original com as novas colunas de predição
df_final = pd.concat([df_novos_dados, df_predictions], axis=1)

# Remove a coluna de texto processado que foi usada apenas internamente
if 'texto_processado' in df_final.columns:
    df_final = df_final.drop(columns=['texto_processado'])

# Salva o resultado final em um novo arquivo Excel
try:
    df_final.to_excel(NOME_ARQUIVO_SAIDA, index=False)
    print(f"-> Arquivo final '{NOME_ARQUIVO_SAIDA}' salvo com sucesso!")
except Exception as e:
    sys.exit(f"ERRO CRÍTICO ao salvar o arquivo de saída: {e}")

print("\n--- PROCESSO DE CLASSIFICAÇÃO FINALIZADO ---")