import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os


def powerset(s): #Essa função gera o powerset de uma lista (excluindo o conjunto vazio)
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1, 1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]



os.chdir("/mnt/HD/R/ml/tpi1") #Colocar diretorio do arquivo prostate.data

df = pd.read_csv("prostate.data", sep='\t')
df = df.drop(['rain', "%"], axis=1) #A coluna "rain" e vazia e o indice, desnecessario

dfx = df['lpsa'] #Target
colunas = ['lcavol', 'lweigh', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
dfy = df[colunas] #Features
pscol = powerset(colunas) #Todas as regressões lineares possiveis com esses dados
resultados = {}

for par in pscol: #Loop calcula todas as regressões
    pdfy = dfy[par]
    xtrain, xtest, ytrain, ytest = train_test_split(pdfy, dfx, test_size=0.3, random_state=0)# Divide em variaveis de treinamento e de teste
    mod = LinearRegression().fit(xtrain, ytrain)
    resultados[" ".join(str(x) for x in par)] = mod.score(xtest, ytest)#Cria um dict com o score (R2) de todas as regressões contra o os conjuntos de teste


variaveis = max(resultados, key=resultados.get).split() #O modelo com o melhor resultado (R2=0,65)


# Create standardizer
standardizer = StandardScaler()

# Standardize features
sdfy = pd.DataFrame(standardizer.fit_transform(dfy))

xtrain, xtest, ytrain, ytest = train_test_split(sdfy, dfx, test_size=0.3, random_state=1)

for i in range(1, 15):# Cria regressões KNN com 1 a 15 vizinhos mais proximos
    reg = KNeighborsRegressor(n_neighbors=i)
    modknn = reg.fit(xtrain, ytrain)
    print(modknn.score(xtest, ytest)) #Nenhum dos modelos obteve resultado melhor que R2=0,35.Escolhe-se as regressoes lineares.


dfy = df[variaveis] #['lcavol', 'lweigh', 'age', 'lbph', 'svi', 'pgg45']

xtrain, xtest, ytrain, ytest = train_test_split(dfy, dfx, test_size=0.3, random_state=0) #Calcula os coeficientes do modelo
modelo = LinearRegression().fit(xtrain, ytrain)

print(variaveis) #['lcavol', 'lweigh', 'age', 'lbph', 'svi', 'pgg45']

print(modelo.coef_)#[ 0.5549137   0.39678239 -0.02150014  0.09880529  0.57475641  0.0032435 ]

print (modelo.intercept_)#1.4005636234119434



