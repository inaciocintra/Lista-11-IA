import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
import warnings
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# *****depois de rodar o codigo, ao abrir uma janela gráfica feche ela ou restante do código não executará até ela ser fechada*****

#ignorar warning 
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

#carregar 
df_treino = pd.read_csv('train.csv')

#pré-processamento
df_treino['Age'] = df_treino['Age'].fillna(df_treino['Age'].median())
df_treino['Embarked'] = df_treino['Embarked'].fillna(df_treino['Embarked'].mode()[0])
df_treino['Fare'] = df_treino['Fare'].fillna(df_treino['Fare'].median())
df_treino['Title'] = df_treino['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df_treino = pd.get_dummies(df_treino, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

#seleção das features
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'] + \
           [col for col in df_treino.columns if col.startswith('Sex_') or 
                                               col.startswith('Embarked_') or
                                               col.startswith('Title_')]

X = df_treino[features]
y = df_treino['Survived']

#divide em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

#arvore
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, None],
    'max_features': [0.8, 'sqrt', None]
}

grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_treino, y_treino)

print("Árvore de Decisão - Melhores hiperparâmetros:", grid.best_params_)
print("Árvore de Decisão - Melhor score:", grid.best_score_)

#avaliação Árvore 
modelo_final = grid.best_estimator_
y_pred = modelo_final.predict(X_teste)
print("\nÁrvore de Decisão - Avaliação")
print("Acurácia:", accuracy_score(y_teste, y_pred))
print("Matriz de confusão:\n", confusion_matrix(y_teste, y_pred))
print("Relatório de classificação:\n", classification_report(y_teste, y_pred))
#visualização
cm = ConfusionMatrix(modelo_final)
cm.fit(X_treino, y_treino)
cm.score(X_teste, y_teste)
cm.show()








#random forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_treino, y_treino)
y_pred_rf = rf.predict(X_teste)
print("\nRandom Forest - Avaliação")
print("Acurácia:", accuracy_score(y_teste, y_pred_rf))
print("Matriz de confusão:\n", confusion_matrix(y_teste, y_pred_rf))
print("Relatório de classificação:\n", classification_report(y_teste, y_pred_rf))
#visualização
cm_rf = ConfusionMatrix(rf)
cm_rf.fit(X_treino, y_treino)
cm_rf.score(X_teste, y_teste)
cm_rf.show()







#3. agrupamento

#aplicando KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

#adicionando os clusters
df_treino['Cluster'] = clusters
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
#plotando clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette='Set1')
plt.title('Clusters de passageiros (KMeans) com PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()
#análise dos clusters
print("\nAnálise dos Clusters:")
print(df_treino.groupby('Cluster')[['Survived', 'Sex_male', 'Pclass']].mean())






#4.asociação 
#importanto 
from mlxtend.frequent_patterns import apriori, association_rules
#criando variáveis binárias paraassociação
df_assoc = df_treino.copy()
df_assoc['Sex'] = np.where(df_assoc['Sex_male']==1, 'male', 'female')
df_assoc['Class'] = df_assoc['Pclass'].astype(str)
df_assoc['Survived'] = df_assoc['Survived'].astype(str)

#selecionando colunas
df_assoc = df_assoc[['Sex', 'Class', 'Survived']]
#transformação para transações
df_onehot = pd.get_dummies(df_assoc)
frequent_itemsets = apriori(df_onehot, min_support=0.1, use_colnames=True)

#regras de associação
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

#mostrando as 3 principais regras
print("\nRegras de Associação:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(3))

