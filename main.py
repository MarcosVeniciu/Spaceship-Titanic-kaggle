import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
# Importa a função de entropia de informação
from scipy.stats import entropy

# Carregando os dados
train_data = pd.read_csv("spaceship-titanic/train.csv")
test_data = pd.read_csv("spaceship-titanic/test.csv")
print("\n\n")
print(train_data.head(9))
print()
##### processamento dos dados

# Verifica a quantidade de registros no dataset
print("Total de registros: " + str(len(train_data)))
print()

# Verifica a quantidade de valores Null presentes no dataset
print("Quantidade de registros com valores Null.")
print(pd.isnull(train_data).sum())

# remove os registros com valores nulos
new_train_data = train_data.dropna()
new_test_data = test_data.dropna()

print("\nQuantidade de registros sem valores null: " + str(len(new_train_data)))
print("Diferença de " + str(len(train_data) - len(new_train_data)) + " registros.")


print("\n\nNome das variaveis:")
print(new_train_data.dtypes)

# Altera os valores boleanos de True e False para 1 e 0.
new_train_data[['CryoSleep', 'VIP', 'Transported']] = new_train_data[['CryoSleep', 'VIP', 'Transported']].replace({True: 1, False: 0})
new_test_data[['CryoSleep', 'VIP']] = new_test_data[['CryoSleep', 'VIP']].replace({True: 1, False: 0})


## Ajustes nas variaveis
# PassengerId
pattern = r"(\d{4})_(\d{2})"

new_train_data["Grupo"] = new_train_data["PassengerId"].str.extract(pattern)[0]
new_train_data["Passageiro_Numero"] = new_train_data["PassengerId"].str.extract(pattern)[1]
new_train_data.drop("PassengerId", axis=1, inplace=True)

new_test_data["Grupo"] = new_test_data["PassengerId"].str.extract(pattern)[0]
new_test_data["Passageiro_Numero"] = new_test_data["PassengerId"].str.extract(pattern)[1]
new_test_data.drop("PassengerId", axis=1, inplace=True)

# Cabin
new_train_data[["Deck", "Cabine_Numero", "Lado"]] = new_train_data["Cabin"].str.split("/", expand=True)
new_train_data.drop("Cabin", axis=1, inplace=True)

new_test_data[["Deck", "Cabine_Numero", "Lado"]] = new_test_data["Cabin"].str.split("/", expand=True)
new_test_data.drop("Cabin", axis=1, inplace=True)

# Removendo a coluna com o nome dos passageiros
new_train_data = new_train_data.drop("Name", axis=1)
#new_test_data = new_test_data.drop("Name", axis=1)


# Normalização
colunas_para_normalizar = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

scaler = StandardScaler()
new_train_data[colunas_para_normalizar] = scaler.fit_transform(new_train_data[colunas_para_normalizar])
new_test_data[colunas_para_normalizar] = scaler.fit_transform(new_test_data[colunas_para_normalizar])


# Ajuste do nome dos planetas
lista_planetas = new_train_data['HomePlanet'].unique()
lista_planetas_test = new_test_data['HomePlanet'].unique()

for planeta in lista_planetas:
    new_train_data[planeta] = (new_train_data['HomePlanet'] == planeta).astype('int')

for planeta in lista_planetas_test:
    new_test_data[planeta] = (new_test_data['HomePlanet'] == planeta).astype('int')
    
# Exclui a coluna original 'HomePlanet'
new_train_data.drop('HomePlanet', axis=1, inplace=True)
new_test_data.drop('HomePlanet', axis=1, inplace=True)

# Ajuste para Destination
lista_planeta_destino = new_train_data['Destination'].unique()
lista_planeta_destino_test = new_test_data['Destination'].unique()

for planeta_destino in lista_planeta_destino:
    new_train_data[planeta_destino] = (new_train_data['Destination'] == planeta_destino).astype('int')

for planeta_destino in lista_planeta_destino_test:
    new_test_data[planeta_destino] = (new_test_data['Destination'] == planeta_destino).astype('int')

# Exclui a coluna original 'Destination'
new_train_data.drop('Destination', axis=1, inplace=True)
new_test_data.drop('Destination', axis=1, inplace=True)


# Ajuste para Deck
lista_planeta_destino = new_train_data['Deck'].unique()
for planeta_destino in lista_planeta_destino:
    new_train_data[planeta_destino] = (new_train_data['Deck'] == planeta_destino).astype('int')

lista_planeta_destino_test = new_test_data['Deck'].unique()
for planeta_destino in lista_planeta_destino_test:
    new_test_data[planeta_destino] = (new_test_data['Deck'] == planeta_destino).astype('int')


# Exclui a coluna original 'Deck'
new_train_data.drop('Deck', axis=1, inplace=True)
new_test_data.drop('Deck', axis=1, inplace=True)

# Ajuste para Lado
lista_planeta_destino = new_train_data['Lado'].unique()
for planeta_destino in lista_planeta_destino:
    new_train_data[planeta_destino] = (new_train_data['Lado'] == planeta_destino).astype('int')

lista_planeta_destino_test = new_test_data['Lado'].unique()
for planeta_destino in lista_planeta_destino_test:
    new_test_data[planeta_destino] = (new_test_data['Lado'] == planeta_destino).astype('int')


# Exclui a coluna original 'Lado'
new_train_data.drop('Lado', axis=1, inplace=True)
new_test_data.drop('Lado', axis=1, inplace=True)

new_train_data['Grupo'] = new_train_data['Grupo'].astype(int)
new_test_data['Grupo'] = new_test_data['Grupo'].astype(int)

new_train_data['Cabine_Numero'] = new_train_data['Cabine_Numero'].astype(int)
new_test_data['Cabine_Numero'] = new_test_data['Cabine_Numero'].astype(int)

new_train_data['Passageiro_Numero'] = new_train_data['Passageiro_Numero'].astype(int)
new_test_data['Passageiro_Numero'] = new_test_data['Passageiro_Numero'].astype(int)


# Treinamento

print("\n\nDados de Treinamento:")
print(new_train_data.head(9))

print("\n\n\n\nInicio do Treinamento...\n")
clf = RandomForestClassifier(n_estimators=100, max_depth=15)

new_train_data_X = new_train_data.drop('Transported', axis=1)
new_train_data_Y = new_train_data['Transported']

clf.fit(new_train_data_X, new_train_data_Y)

print("\n\nResultado de treinamento:")
score = clf.score(new_train_data_X, new_train_data_Y)
print("Acurácia:", score)


## Predição dos dados de teste
print("\n\nDados de Teste:")
print(new_test_data.head(9))

X_test = new_test_data.drop('Name', axis=1)
column_order = list(new_train_data_X.columns)

X_test = X_test.reindex(columns=column_order)

y_pred = clf.predict(X_test)

print(len(y_pred))