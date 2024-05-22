import numpy as np

# Data untuk pola AND
X_and = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
)
y_and = np.array(
    [0, 0, 0, 1],
)


bobot_and = np.zeros(X_and.shape[1])
bias_and = 0


for i in range(X_and.shape[0]):
    bobot_and += X_and[i] * y_and[i]
    bias_and += y_and[i]


prediksi_and = np.zeros(X_and.shape[0])
for i in range(X_and.shape[0]):
    hasil = np.dot(X_and[i], bobot_and) + bias_and
    prediksi_and[i] = 1 if hasil > 0 else 0


akurasi_and = np.mean(prediksi_and == y_and) * 100

print(f"Bobot Pola AND: {bobot_and}, Bias: {bias_and}")
print(f"Prediksi Pola AND: {prediksi_and}")
print(f"Akurasi Pola AND: {akurasi_and:.2f}%")


X_3_input = np.array(
    [
        [-1, -1, -1],
        [-1, 1, -1],
        [1, -1, -1],
        [1, 1, 1],
    ]
)
y_3_input = np.array(
    [-1, -1, -1, 1],
)

# Inisialisasi bobot dan bias untuk pola 3-input
bobot_3_input = np.zeros(X_3_input.shape[1])
bias_3_input = 0

# Proses pelatihan pola 3-input
for i in range(X_3_input.shape[0]):
    bobot_3_input += X_3_input[i] * y_3_input[i]
    bias_3_input += y_3_input[i]

# Prediksi pola 3-input
prediksi_3_input = np.zeros(X_3_input.shape[0])
for i in range(X_3_input.shape[0]):
    hasil = np.dot(X_3_input[i], bobot_3_input) + bias_3_input
    prediksi_3_input[i] = 1 if hasil > 0 else 0

# Konversi prediksi bipolar untuk mencocokkan format target (-1, 1)
prediksi_3_input_bipolar = np.where(prediksi_3_input == 0, -1, 1)

# Hitung akurasi pola 3-input
akurasi_3_input = np.mean(prediksi_3_input_bipolar == y_3_input) * 100

print(f"Bobot Pola 3-Input: {bobot_3_input}, Bias: {bias_3_input}")
print(f"Prediksi Pola 3-Input: {prediksi_3_input_bipolar}")
print(f"Akurasi Pola 3-Input: {akurasi_3_input:.2f}%")
