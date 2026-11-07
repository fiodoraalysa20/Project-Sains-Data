# Dashboard Skrining Diabetes (Capstone Project)

Dashboard Streamlit untuk memprediksi risiko diabetes berdasarkan data klinis
(Pima Indians Diabetes Database, Kaggle).

Isi folder:
- app.py                 : kode dashboard
- requirements.txt       : library yang diinstal Streamlit Cloud
- generate_diabetes.pkl  : pipeline scaler + model terbaik (hasil Capstone 2)
- model_report.json      : ringkasan evaluasi model (hasil Capstone 2)
- data.csv               : data bersih (hasil Capstone 1)
- .streamlit/config.toml : tema warna dashboard

Menjalankan di komputer sendiri:
    pip install -r requirements.txt
    streamlit run app.py
