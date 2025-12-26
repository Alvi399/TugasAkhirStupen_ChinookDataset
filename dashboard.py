import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ===============================
# LOAD MODEL & ENCODER
# ===============================
model = joblib.load("./xgboost_model.pkl")
label_encoders = joblib.load("./label_encoders.pkl")

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="Dashboard Prediksi Revenue Musik Digital",
    layout="wide"
)

# ===============================
# TITLE
# ===============================
st.title("📊 Dashboard Prediksi Revenue Musik Digital (XGBoost)")
st.write(
    """
    Dashboard ini digunakan untuk menampilkan hasil prediksi revenue industri musik digital 
    berdasarkan data historis Chinook Database menggunakan model XGBoost.
    """
)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("Pengaturan Prediksi")
tahun_prediksi = st.sidebar.selectbox("Tahun Prediksi", [2026])
unit_price = st.sidebar.number_input("Unit Price", value=0.99)
quantity = st.sidebar.number_input("Quantity", value=1)

# ===============================
# GENERATE FUTURE DATA
# ===============================
future_data = []

for month in range(1, 13):
    for day in range(1, 29):
        future_data.append({
            'unit_price': unit_price,
            'quantity': quantity,
            'tahun_angka': tahun_prediksi,
            'hari_dalam_bulan_angka': day,
            'bulan': month,
            'genre_name': 17,          # asumsi
            'billing_country': 11      # asumsi
        })

future_df = pd.DataFrame(future_data)

# ===============================
# ENCODING
# ===============================
for col in ['bulan', 'genre_name', 'billing_country']:
    known_classes = set(label_encoders[col].classes_)
    future_df[col] = future_df[col].apply(
        lambda x: label_encoders[col].transform([x])[0] if x in known_classes else 0
    )

# ===============================
# PREDICTION
# ===============================
future_df['predicted_total_revenue'] = model.predict(future_df)

# ===============================
# METRICS
# ===============================
total_revenue = future_df['predicted_total_revenue'].sum()

st.metric(
    label="💰 Total Predicted Revenue Tahun 2026",
    value=f"{total_revenue:,.2f}"
)

# ===============================
# TABS
# ===============================
tab1, tab2, tab3 = st.tabs(["📈 Tren Revenue", "📄 Data Prediksi", "🧠 Insight Bisnis"])

# ===============================
# TAB 1 - TREND
# ===============================
with tab1:
    st.subheader("Tren Prediksi Revenue Bulanan")

    monthly_revenue = (
        future_df
        .groupby("bulan")["predicted_total_revenue"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots()
    ax.plot(
        monthly_revenue["bulan"],
        monthly_revenue["predicted_total_revenue"],
        marker="o"
    )
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Total Predicted Revenue")
    ax.set_title("Prediksi Revenue Bulanan Tahun 2026")

    st.pyplot(fig)

# ===============================
# TAB 2 - TABLE
# ===============================
with tab2:
    st.subheader("Tabel Hasil Prediksi")
    st.dataframe(
        future_df[['tahun_angka', 'bulan', 'hari_dalam_bulan_angka', 'predicted_total_revenue']]
        .head(100)
    )

# ===============================
# TAB 3 - INSIGHT
# ===============================
with tab3:
    st.subheader("Insight Bisnis")

    st.write(
        """
        Berdasarkan hasil prediksi, revenue industri musik digital menunjukkan variasi 
        antar bulan pada tahun 2026. Pola ini mengindikasikan bahwa faktor waktu memiliki 
        pengaruh terhadap performa penjualan, meskipun variabel lain diasumsikan konstan.

        Insight ini dapat digunakan sebagai dasar dalam:
        - Menentukan waktu promosi yang optimal
        - Perencanaan strategi distribusi konten musik
        - Evaluasi potensi pasar musik digital di masa depan
        """
    )

# ===============================
# FOOTER
# ===============================
st.caption("Studi Independen – Business Intelligence & Artificial Intelligence")
