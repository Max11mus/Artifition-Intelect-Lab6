# Імпортуємо необхідні бібліотеки
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
import numpy as np
import matplotlib.pyplot as plt

# Генеруємо дані для навчання моделі
# Вхідні дані x - значення від -10 до 10 з кроком 0.2
x_train = np.arange(-10,10.2,0.2)
# Вихідні дані y - значення 2^x
y_train = 2**x_train

# Створюємо модель нейромережі
# Вхідний шар з одним нейроном
input_layer = Input(shape=(1,))
# Прихований шар з 10 нейронами та функцією активації сігмоїдального типу
hidden_layer = Dense(10, activation='sigmoid')(input_layer)
# Вихідний шар з одним нейроном
output_layer = Dense(1)(hidden_layer)

# Створюємо модель нейромережі
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

# Виводимо опис моделі
model.summary()

# Компілюємо модель, встановлюючи функцію втрат та оптимізатор
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1))

# Навчаємо модель на вхідних даних
model.fit(x = x_train, y = y_train, epochs=10000, batch_size=101)

# Проводимо апроксимацію на тестових даних навчання
# Виконуємо передбачення за допомогою моделі
x_test = np.arange(-10.1,10.2,0.2)
y_test = model.predict(x_test)

# Виводимо результати
plt.scatter(x_train, y_train, s=1, color='blue', label='Дані для навчання')
plt.scatter(x_test, y_test, s=1, color='red', label='Апроксимація')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Апроксимація функції y=2^x за допомогою нейромережі')
plt.savefig(fname='aproximation.png', dpi=300)
