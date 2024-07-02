import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Import data from CSV
file_path = 'waktu_belajar_nilai_siswa.csv'
df = pd.read_csv(file_path)

# Load the data
X = df[['Waktu Belajar']].values
y = df['Nilai Siswa'].values

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plot the data and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data Siswa')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regresi Linier')
plt.title('Regresi Linier Sederhana')
plt.xlabel('Waktu Belajar (jam)')
plt.ylabel('Nilai Siswa')
plt.legend()
plt.grid(True)
plt.show()
