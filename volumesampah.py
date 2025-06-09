import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

st.set_page_config(page_title="Prediksi Total Sampah Kota Sukabumi", layout="wide")

st.title("♻️ Prediksi Total Sampah Kota Sukabumi")
st.markdown("Model prediksi menggunakan **Linear Regression**, **Polynomial Regression (degree 2)**, dan **XGBoost** berdasarkan data tahunan.")

# === Load Data ===
@st.cache_data
def load_data():
    df = pd.read_csv("data sampah kota sukabumi.csv", sep=";", skiprows=1)
    df.columns = df.columns.astype(str)
    baris_tahunan = df[df['BULAN'].str.upper().str.strip() == 'TAHUNAN']
    data_tahun = baris_tahunan.drop(columns=['BULAN']).T.reset_index()
    data_tahun.columns = ['Tahun', 'Total_Sampah']
    data_tahun['Tahun'] = data_tahun['Tahun'].astype(int)
    data_tahun['Total_Sampah'] = data_tahun['Total_Sampah'].astype(float)
    return data_tahun

data_tahun = load_data()

X = data_tahun[['Tahun']]
y = data_tahun['Total_Sampah']

# Buat data tahun prediksi tambahan (extend ke 2027)
tahun_min = data_tahun['Tahun'].min()
tahun_max = data_tahun['Tahun'].max()
tahun_pred_all = pd.DataFrame({'Tahun': list(range(tahun_min, 2028))})

# === Model Linear Regression ===
linreg = LinearRegression()
linreg.fit(X, y)
y_lin_pred = linreg.predict(X)
y_lin_future = linreg.predict(tahun_pred_all)

# === Model Polynomial Regression (degree=2) ===
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
tahun_pred_poly = poly.transform(tahun_pred_all)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)
y_poly_future = poly_model.predict(tahun_pred_poly)

# === Model XGBoost Regressor ===
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(X, y)
y_xgb_pred = xgb_model.predict(X)
y_xgb_future = xgb_model.predict(tahun_pred_all)

# Fungsi evaluasi model
def evaluate_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

mse_lin, r2_lin = evaluate_model("Linear Regression", y, y_lin_pred)
mse_poly, r2_poly = evaluate_model("Polynomial Regression", y, y_poly_pred)
mse_xgb, r2_xgb = evaluate_model("XGBoost Regressor", y, y_xgb_pred)

# Tampilkan skor akurasi
st.subheader("📊 Evaluasi Model pada Data Historis")
st.write(f"Linear Regression — MSE: {mse_lin:.2f}, R²: {r2_lin:.4f}")
st.write(f"Polynomial Regression — MSE: {mse_poly:.2f}, R²: {r2_poly:.4f}")
st.write(f"XGBoost Regressor — MSE: {mse_xgb:.2f}, R²: {r2_xgb:.4f}")

# Input tahun untuk prediksi
st.subheader("🔮 Prediksi Total Sampah per Tahun")
tahun_input = st.number_input(
    "Masukkan Tahun untuk Prediksi", 
    min_value=tahun_min, max_value=2030, value=tahun_max + 1, step=1)

tahun_input_df = pd.DataFrame({'Tahun': [tahun_input]})
tahun_input_poly = poly.transform(tahun_input_df)

pred_lin = linreg.predict(tahun_input_df)[0]
pred_poly = poly_model.predict(tahun_input_poly)[0]
pred_xgb = xgb_model.predict(tahun_input_df)[0]

st.markdown(f"### Hasil Prediksi Tahun {tahun_input}:")
st.write(f"- Linear Regression: **{pred_lin:.2f} ton**")
st.write(f"- Polynomial Regression (degree 2): **{pred_poly:.2f} ton**")
st.write(f"- XGBoost Regressor: **{pred_xgb:.2f} ton**")

# Visualisasi
st.subheader("📈 Grafik Perbandingan Model")

fig, ax = plt.subplots(figsize=(12, 6))

# Plot data asli
ax.scatter(X, y, color='black', label='Data Asli')

# Plot prediksi model historis
ax.plot(tahun_pred_all, y_lin_future, linestyle='--', color='red', label='Linear Regression')
ax.plot(tahun_pred_all, y_poly_future, linestyle='--', color='green', label='Polynomial Regression')
ax.plot(tahun_pred_all, y_xgb_future, linestyle='--', color='blue', label='XGBoost Regressor')

# Highlight prediksi tahun input user dengan warna berbeda
colors = ['red', 'green', 'blue']
preds = [pred_lin, pred_poly, pred_xgb]
names = ['Linear', 'Polynomial', 'XGBoost']

for pred, color, name in zip(preds, colors, names):
    ax.scatter(tahun_input, pred, color=color, s=100, zorder=5)
    ax.annotate(f"{pred:.1f}", (tahun_input, pred), textcoords="offset points", xytext=(0,10), ha='center', color=color)

ax.set_xlabel('Tahun')
ax.set_ylabel('Total Sampah (ton)')
ax.set_title('Prediksi Total Sampah Kota Sukabumi per Tahun')
ax.grid(True)
ax.legend()
st.pyplot(fig)
