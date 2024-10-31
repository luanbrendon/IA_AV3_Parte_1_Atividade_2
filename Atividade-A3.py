import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Carregar os dados
df = pd.read_csv("Dados_RH_Turnover.csv", delimiter=";")

# Separar as variáveis explicativas (X) e a variável alvo (y)
X = df.drop(columns=['SaiuDaEmpresa', 'DeptoAtuacao', 'Salario'])
y = df['SaiuDaEmpresa']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalonar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dicionário para armazenar os modelos e seus resultados
models = {
    "Regressão Logística": LogisticRegression(max_iter=200),
    "Árvore de Decisão": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Rede Neural (MLP)": MLPClassifier(max_iter=200)
}

# Treinar e avaliar cada modelo
for model_name, model in models.items():
    # Treinar o modelo
    model.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred = model.predict(X_test)
    
    # Calcular a acurácia e a matriz de confusão
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Exibir os resultados
    print(f"{model_name} - Acurácia: {accuracy}")
    print("Matriz de Confusão:")
    print(conf_matrix)
    print("\n" + "="*40 + "\n")
