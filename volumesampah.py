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
    df = pd.read_csv('data sampah kota sukabumi.csv', sep=';', skiprows=1)
    return df

df = load_data()

# === Validasi Bulan ===
bulan_valid = [
    "JANUARI", "FEBRUARI", "MARET", "APRIL", "MEI", "JUNI",
    "JULI", "AGUSTUS", "SEPTEMBER", "OKTOBER", "NOVEMBER", "DESEMBER"
]
df = df[df["BULAN"].isin(bulan_valid)].reset_index(drop=True)

# === Konversi nama bulan ke angka ===
bulan_map = {
    'JANUARI': 1, 'FEBRUARI': 2, 'MARET': 3, 'APRIL': 4,
    'MEI': 5, 'JUNI': 6, 'JULI': 7, 'AGUSTUS': 8,
    'SEPTEMBER': 9, 'OKTOBER': 10, 'NOVEMBER': 11, 'DESEMBER': 12
}
df['BULAN'] = df['BULAN'].replace(bulan_map)

# === Fitur dan Target ===
x = df['BULAN'].values.reshape(-1, 1)
tahun_list = ['2017', '2018', '2020', '2021', '2022', '2023']
y = df[tahun_list].values

# === Model Training ===
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)

# === Input Pengguna ===
st.subheader("🔮 Prediksi Volume Sampah Bulanan")
col1, col2 = st.columns(2)
with col1:
    bulan_input = st.selectbox("Pilih Bulan", list(bulan_map.keys()))
with col2:
    tahun_prediksi_input = st.number_input("Masukkan Tahun Prediksi", min_value=2024, max_value=2100, step=1, value=2024)

bulan_numerik = bulan_map[bulan_input]
prediksi = model.predict([[bulan_numerik]])

# Tampilkan hasil prediksi
st.markdown("### Hasil Prediksi:")
prediksi_dict = {}
for i, thn in enumerate(tahun_list):
    st.write(f"📅 Bulan **{bulan_input}** Tahun **{thn}** → **{prediksi[0][i]:.2f} ton**")
    prediksi_dict[thn] = prediksi[0][i]

# Tambahkan hasil prediksi tahun input
prediksi_dict[str(tahun_prediksi_input)] = np.mean(prediksi[0])

st.write(f"📅 Bulan **{bulan_input}** Tahun **{tahun_prediksi_input}** (prediksi) → **{prediksi_dict[str(tahun_prediksi_input)]:.2f} ton**")

# === Skor Model ===
st.subheader("📊 Skor Akurasi Model")
skor = model.score(x_test, y_test)
st.write(f"Skor R²: **{skor:.4f}**")

# === Visualisasi ===
st.subheader("📈 Volume Sampah + Prediksi (Tahun Prediksi)")
bulan_label = df['BULAN'].replace({v: k for k, v in bulan_map.items()})
x_bar = np.arange(len(bulan_label))
bar_width = 0.12

fig, ax = plt.subplots(figsize=(14, 6))
for i, thn in enumerate(tahun_list):
    posisi = x_bar + i * bar_width
    ax.bar(posisi, df[thn], width=bar_width, label=thn)

# Prediksi tahun input
posisi_pred = x_bar + len(tahun_list) * bar_width
prediksi_repeat = [prediksi_dict[str(tahun_prediksi_input)]] * len(x_bar)
ax.bar(posisi_pred, prediksi_repeat, width=bar_width, label=f'{tahun_prediksi_input} (Prediksi)', color='red', alpha=0.6)

ax.set_xlabel("Bulan")
ax.set_ylabel("Volume Sampah (ton)")
ax.set_title(f"Volume Sampah Kota Sukabumi per Bulan + Prediksi Tahun {tahun_prediksi_input}")
ax.set_xticks(x_bar + bar_width * (len(tahun_list) / 2))
ax.set_xticklabels(bulan_label, rotation=45)
ax.legend()

st.pyplot(fig)
