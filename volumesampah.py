import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

st.set_page_config(page_title="Prediksi Sampah Kota Sukabumi", layout="wide")

st.title("🗑️ Prediksi Volume Sampah Kota Sukabumi")
st.markdown("Model prediksi menggunakan **Linear Regression** berdasarkan data dari tahun 2017–2023.")

# === Load Data ===
@st.cache_data
def load_data():
    return pd.read_csv("data sampah kota sukabumi.csv", sep=";")

df = load_data()

# === Validasi Bulan ===
bulan_map = {
    'JANUARI': 1, 'FEBRUARI': 2, 'MARET': 3, 'APRIL': 4,
    'MEI': 5, 'JUNI': 6, 'JULI': 7, 'AGUSTUS': 8,
    'SEPTEMBER': 9, 'OKTOBER': 10, 'NOVEMBER': 11, 'DESEMBER': 12
}
bulan_map_reverse = {v: k for k, v in bulan_map.items()}

# Bersihkan dan ubah bulan ke angka
df = df[df['BULAN'].isin(bulan_map.keys())].copy()
df['BULAN'] = df['BULAN'].map(bulan_map)

# Buat fitur waktu gabungan: TAHUN * 100 + BULAN
df['WAKTU'] = df['TAHUN'].astype(int) * 100 + df['BULAN']

# Fitur dan target
x = df[['WAKTU']].values
y = df['VOLUME'].values

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(x_train, y_train)

# === Input Prediksi ===
st.subheader("🔮 Prediksi Volume Sampah Bulanan")

col1, col2 = st.columns(2)
with col1:
    bulan_input = st.selectbox("Pilih Bulan", list(bulan_map.keys()))
with col2:
    tahun_input = st.number_input("Masukkan Tahun", min_value=2024, max_value=2100, step=1, value=2024)

bulan_numerik = bulan_map[bulan_input]
waktu_prediksi = tahun_input * 100 + bulan_numerik
volume_pred = model.predict([[waktu_prediksi]])[0]

# Tampilkan hasil
st.markdown(f"### 📅 Prediksi Bulan **{bulan_input} {tahun_input}** → **{volume_pred:.2f} ton**")

# === Skor Akurasi ===
st.subheader("📊 Skor Akurasi Model")
score = model.score(x_test, y_test)
st.write(f"Skor R²: **{score:.4f}**")

# === Visualisasi ===
st.subheader("📈 Visualisasi Volume Sampah")

fig, ax = plt.subplots(figsize=(12, 5))
# Plot data historis
df_sorted = df.sort_values(by='WAKTU')
ax.plot(df_sorted['WAKTU'], df_sorted['VOLUME'], label="Data Historis", marker='o')

# Tambahkan prediksi
ax.scatter(waktu_prediksi, volume_pred, color='red', label="Prediksi", s=100, zorder=5)
ax.annotate(f"{volume_pred:.1f} ton", (waktu_prediksi, volume_pred),
            textcoords="offset points", xytext=(0,10), ha='center', color='red')

# Format x-axis sebagai TAHUN-BULAN
xticks = df_sorted['WAKTU'].unique()
xtick_labels = [f"{str(w)[:4]}-{bulan_map_reverse[int(str(w)[4:])]}" for w in xticks]
ax.set_xticks(xticks[::2])
ax.set_xticklabels(xtick_labels[::2], rotation=45)

ax.set_xlabel("Waktu (Tahun-Bulan)")
ax.set_ylabel("Volume Sampah (ton)")
ax.set_title("Volume Sampah Kota Sukabumi dan Prediksi")
ax.legend()
st.pyplot(fig)
