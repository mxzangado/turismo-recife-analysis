#!/usr/bin/env python
# coding: utf-8

# <h1>1. Carregamento e Inspeção Inicial</h1>

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Carregamento do dataset de avaliações turísticas em Recife
df = pd.read_csv(r'D:\Estatistica\avaliacao_turistas_recife (2).csv')


# ## Avaliação turistica Recife

# In[3]:


# primeiras linhas para entender as colunas
df


# In[4]:


# Verificação de tipos de dados e valores nulos
# O dataset tem 200 linhas e 7 colunas (int64 e object)
df.info()


# <h1>2. Limpeza de Dados (Data Cleaning)</h1>

# In[5]:


# 1. Limpa os nomes das colunas (remove espaços invisíveis)
df.columns = df.columns.str.strip()
colunas_notas = ['Avaliacao_Atrativos', 'Avaliacao_Hospedagem', 'Avaliacao_Transporte', 'Avaliacao_Gastronomia']
# Criar uma coluna de 'Média Geral' de satisfação por turista
df['Media_satisfacao'] = df[colunas_notas].mean(axis=1)
# axis =1 A operação é feita horizontalmente (ao longo das colunas), calculando a média para cada linha.


# In[6]:


# Visualizando a criação da coluna Media satisfação
df.info()


# ## Análise de Desempenho por Categoria
# **Pergunta de Negócio:** Qual dos serviços turísticos em Recife (Atrativos, Hospedagem, Transporte ou Gastronomia) possui a menor nota média e precisa de maior atenção?
# 
# > **Hipótese:** O transporte pode ter notas menores devido ao trânsito intenso em horários de pico na região do Marco Zero e Boa Viagem.
# 3. Engenharia de Atributos (Feature Engineering)
# Criação de uma nova métrica para resumir a experiência do turista.

# <h3>"Existe uma diferença significativa na satisfação entre turistas brasileiros e estrangeiros?"</h3>

# In[7]:


df_media_nacionalidade = df.groupby('Nacionalidade')['Media_satisfacao'].mean().sort_values(ascending=False)
print(df_media_nacionalidade)


# In[8]:


# 1. Definir o tamanho da figura
plt.figure(figsize=(10, 6))

# 2. Criar o gráfico de barras
# .index são os nomes dos países, .values são as médias
plt.bar(df_media_nacionalidade.index, df_media_nacionalidade.values, color='skyblue')

# 3. Adicionar títulos e nomes nos eixos
plt.title('Média de Satisfação por Nacionalidade em Recife')
plt.xlabel('Nacionalidade')
plt.ylabel('Nota Média (0 a 5)')

# 4. Rotacionar os nomes dos países para não ficarem amontoados
plt.xticks(rotation=45)

# 5. Definir o limite do eixo Y para a escala real das notas
plt.ylim(0, 5)

# 6. Exibir o gráfico
plt.show()


# <h3>"Turistas que ficam mais tempo."</h3>

# In[9]:


plt.figure(figsize=(8, 5))
sns.regplot(data=df, x='Dias_Estadia', y='Media_satisfacao', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Dispersão: Influência dos Atrativos na satisfação Geral na Estadia')
plt.show()


# <h2>Através do gráfico de dispersão, observamos que a satisfação média permanece estável, aumenta e diminui conforme o turista estende sua permanência em Recife.</h2>

# <h1>O Poder dos Atrativos</h1>

# In[10]:


# Correlação
correlacao_valor = df['Avaliacao_Atrativos'].corr(df['Media_satisfacao'])
print(f"A correlação Avaliacao Atrativos e media satisfação é: {correlacao_valor:.2f}")


# In[11]:


plt.figure(figsize=(8, 5))
sns.regplot(data=df, x='Avaliacao_Atrativos', y='Media_satisfacao', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Dispersão: Correlação Atrativos vs Satisfação ')
plt.show()


# <h3>Qual a correlação entre a nota de Gastronomia e a nota de Hospedagem?</h3>

# In[12]:


# Correlação
correlacao_valor = df['Avaliacao_Hospedagem'].corr(df['Avaliacao_Gastronomia'])
print(f"A correlação entre Hospedagem e Gastronomia é: {correlacao_valor:.2f}")


# In[13]:


plt.figure(figsize=(8, 5))
sns.regplot(data=df, x='Avaliacao_Hospedagem', y='Avaliacao_Gastronomia', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Dispersão: Hospedagem vs Gastronomia')
plt.show()


# <h3>A satisfação com o hotel não tem nada a ver com a comida (são independentes).</h3>

# <h3>A análise demonstra que a qualidade dos pontos turísticos é o motor principal da satisfação em Recife ($r = 0.48$).
# Investimentos em infraestrutura turística terão um impacto muito mais direto na nota da cidade do que melhorias isoladas em hotelaria ou transporte.<br>
# Conclusão: Isso indica que, para o turista em Recife, a qualidade dos pontos turísticos (como o Recife Antigo ou as praias)
# impacta muito mais o seu humor final do que o hotel ou o transporte. Investir na manutenção dos atrativos é a forma mais garantida 
# de elevar a nota da cidade.</h3>
