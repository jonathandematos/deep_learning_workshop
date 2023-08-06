import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Criacao dos datasets sinteticos de teste
X_train, Y_train = make_moons(n_samples=100, shuffle=True, noise=0.3, random_state=0)
X_test, Y_test = make_moons(n_samples=50, shuffle=True, noise=0.3, random_state=1)

# Criacao do plot
figure = plt.figure(figsize=(20, 20))

# Plot dos dados de treino
cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
ax = plt.subplot(2,2,1)
ax.set_title("Treino")
ax.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=cm_bright, edgecolors="k")

# Plot dos dados de treino
ax = plt.subplot(2,2,3)
ax.set_title("Teste")
ax.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap=cm_bright, edgecolors="k")


# Criacao do MLP com 10 neuronios na camada oculta
mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=100, verbose=True, learning_rate='constant', learning_rate_init=0.001)

# Treino do MLP
mlp.fit(X_train, Y_train)

# Inferencia dos dados
Y_train_pred = mlp.predict(X_train)
Y_test_pred = mlp.predict(X_test)

# Relatorio de precisao
print("Classification Report Train")
print(classification_report(Y_train, Y_train_pred))
print("Classification Report Test")
print(classification_report(Y_test, Y_test_pred))

# Verificacao dos dados classificados corretamente
correct_train = (Y_train_pred == Y_train).astype("int")
correct_test = (Y_test_pred == Y_test).astype("int")

# Plot dos dados de acordo com a classificacao correta ou nao
cm = plt.cm.RdBu
cm_bright = ListedColormap(["#000000", "#00FF00"])
ax = plt.subplot(2,2,2)
ax.set_title("TR Pred")
ax.scatter(X_train[:, 0], X_train[:, 1], c=correct_train, cmap=cm_bright, edgecolors="k")

ax = plt.subplot(2,2,4)
ax.set_title("TR Pred")
ax.scatter(X_test[:, 0], X_test[:, 1], c=correct_test, cmap=cm_bright, edgecolors="k")

plt.tight_layout()
plt.show()

exit(0)
