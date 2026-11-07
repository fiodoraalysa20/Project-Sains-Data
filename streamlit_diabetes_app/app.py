import json
import os
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Konfigurasi halaman (harus menjadi perintah Streamlit pertama)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Skrining Diabetes", page_icon="🩸", layout="wide")

INK, BLUE, AMBER, TEAL, LINE, MUTED = "#14233A", "#1F5FAF", "#C77F00", "#1E8A7A", "#D6DDE6", "#5B6B7F"
FEATURES = ['Glucose', 'BMI', 'Age', 'Pregnancies', 'DiabetesPedigreeFunction', 'BloodPressure']
LABEL = {'Glucose': 'Glukosa 2 jam (mg/dl)', 'BMI': 'BMI (kg/m²)', 'Age': 'Usia (tahun)',
         'Pregnancies': 'Jumlah kehamilan', 'DiabetesPedigreeFunction': 'Riwayat keluarga (DPF)',
         'BloodPressure': 'Tekanan darah diastolik (mmHg)'}
# Rentang = rentang data training setelah pembersihan di Capstone 1
RANGE = {'Pregnancies': (0, 13, 3, 1), 'Glucose': (44, 198, 116, 1), 'BloodPressure': (44, 104, 72, 1),
         'BMI': (18.2, 50.0, 32.1, 0.1), 'DiabetesPedigreeFunction': (0.078, 1.191, 0.355, 0.001),
         'Age': (21, 66, 29, 1)}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, input, button, textarea, select {{
    font-family: 'Public Sans', -apple-system, 'Segoe UI', Roboto, sans-serif;
}}
h1, h2, h3 {{ font-family: 'Public Sans', sans-serif; color: {INK}; letter-spacing: -0.015em; }}
h1 {{ font-weight: 800; font-size: 2.1rem; line-height: 1.15; }}
h2 {{ font-weight: 700; font-size: 1.35rem; }}
h3 {{ font-weight: 650; font-size: 1.05rem; }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
.lede {{ color: {MUTED}; font-size: 1.02rem; max-width: 62ch; margin-top: -0.4rem; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {LINE}; }}
[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; font-weight: 700; }}
/* Panel hasil: satu-satunya elemen "berani" di dashboard */
.reading {{ border: 1px solid {LINE}; border-left: 6px solid var(--tone); background: #FFFFFF;
            padding: 1.4rem 1.6rem 1.2rem; border-radius: 4px; }}
.reading .lbl {{ color: {MUTED}; font-size: 0.9rem; margin: 0; }}
.reading .num {{ font-size: 4.2rem; font-weight: 800; line-height: 1; color: var(--tone);
                 font-variant-numeric: tabular-nums; margin: 0.35rem 0 0.2rem; letter-spacing: -0.03em; }}
.reading .num small {{ font-size: 1.6rem; font-weight: 700; margin-left: 0.15rem; }}
.reading .verdict {{ font-size: 1.15rem; font-weight: 700; color: {INK}; margin: 0.2rem 0 0.9rem; }}
.scale {{ position: relative; height: 12px; border-radius: 6px; margin: 0.4rem 0 2.6rem;
          background: linear-gradient(90deg, {TEAL} 0%, #E9E3C9 45%, {AMBER} 100%); }}
.scale .pin {{ position: absolute; top: -6px; width: 4px; height: 24px; background: {INK};
               border-radius: 2px; transform: translateX(-2px); }}
.scale .tick {{ position: absolute; top: 14px; font-size: 0.75rem; color: {MUTED};
                transform: translateX(-50%); white-space: nowrap; }}
.scale .tick::before {{ content: ""; position: absolute; left: 50%; top: -16px; height: 16px;
                        border-left: 1px dashed {MUTED}; }}
.scale .tick.low {{ top: 32px; }}
.scale .tick.low::before {{ top: -34px; height: 34px; }}
.note {{ color: {MUTED}; font-size: 0.86rem; line-height: 1.5; }}
.empty {{ border: 1px dashed {LINE}; border-radius: 4px; padding: 2.2rem 1.6rem; color: {MUTED}; background: #FFFFFF; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pemuatan model dan data (di-cache agar tidak dibaca ulang setiap interaksi)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_diabetes_model():
    with open("generate_diabetes.pkl", "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_data():
    df = pd.read_csv("data.csv", index_col=0)
    df['Status'] = df['Outcome']
    df['Outcome'] = (df['Outcome'] == 'Diabetes').astype(int)
    return df


@st.cache_data
def load_report():
    if not os.path.exists("model_report.json"):
        return None
    with open("model_report.json") as f:
        return json.load(f)


def style_fig(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
                      font=dict(family="Public Sans, sans-serif", color=INK, size=13),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None))
    fig.update_xaxes(gridcolor="#EEF2F6", linecolor=LINE)
    fig.update_yaxes(gridcolor="#EEF2F6", linecolor=LINE)
    return fig


STATUS_COLORS = {'Diabetes': AMBER, 'Non-Diabetes': TEAL}


# ---------------------------------------------------------------------------
# Halaman 1: Skrining pasien
# ---------------------------------------------------------------------------
def fmt(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def reading_panel(proba, youden):
    high = proba >= 0.5
    tone = AMBER if high else TEAL
    verdict = "Perlu pemeriksaan lanjutan" if high else "Risiko rendah"
    if not high and proba >= youden:
        verdict = "Risiko rendah, tetapi dekat ambang batas"
    pct = proba * 100
    st.markdown(f"""
    <div class="reading" style="--tone:{tone}" role="status" aria-live="polite">
      <p class="lbl">Probabilitas diabetes menurut model</p>
      <p class="num">{pct:.0f}<small>%</small></p>
      <p class="verdict">{verdict}</p>
      <div class="scale" aria-hidden="true">
        <div class="tick" style="left:50%">ambang standar 0,50</div>
        <div class="tick low" style="left:{youden*100:.1f}%">ambang sensitif {fmt(youden)}</div>
        <div class="pin" style="left:{min(max(pct, 1), 99):.1f}%"></div>
      </div>
    </div>""", unsafe_allow_html=True)


def page_screening():
    report = load_report()
    youden = report['youden_threshold'].get(report['best_model'], 0.4) if report else 0.4
    st.title("Skrining risiko diabetes")
    st.markdown('<p class="lede">Masukkan hasil pemeriksaan pasien untuk memperkirakan peluang diabetes. '
                'Hasil ini membantu menentukan siapa yang perlu tes lanjutan, bukan diagnosis.</p>',
                unsafe_allow_html=True)

    tab_one, tab_batch = st.tabs(["Satu pasien", "Banyak pasien (CSV)"])
    with tab_one:
        left, right = st.columns([1.05, 1], gap="large")
        with left:
            with st.form("form_pasien", border=False):
                st.subheader("Hasil pemeriksaan")
                c1, c2 = st.columns(2)
                values = {}
                for col, feat in zip([c1, c2, c1, c2, c1, c2],
                                     ['Glucose', 'BloodPressure', 'BMI', 'Age', 'Pregnancies', 'DiabetesPedigreeFunction']):
                    lo, hi, default, step = RANGE[feat]
                    values[feat] = col.number_input(LABEL[feat], min_value=lo, max_value=hi,
                                                    value=default, step=step,
                                                    format="%.3f" if feat == 'DiabetesPedigreeFunction'
                                                    else ("%.1f" if feat == 'BMI' else "%d"))
                submitted = st.form_submit_button("Hitung risiko", type="primary", width='stretch')
            st.markdown('<p class="note">Rentang isian mengikuti data yang dipelajari model. '
                        'Nilai di luar rentang ini tidak bisa diprediksi dengan andal.</p>', unsafe_allow_html=True)
        with right:
            if submitted:
                st.session_state['last_input'] = values
            if 'last_input' not in st.session_state:
                st.markdown('<div class="empty"><strong>Belum ada hasil.</strong><br>'
                            'Isi hasil pemeriksaan di sebelah kiri, lalu tekan <em>Hitung risiko</em>.</div>',
                            unsafe_allow_html=True)
            else:
                patient = pd.DataFrame([st.session_state['last_input']])[FEATURES]
                proba = float(load_diabetes_model().predict_proba(patient)[0, 1])
                reading_panel(proba, youden)
                st.markdown(f'<p class="note">Model: {report["best_model"] if report else "tersimpan"}. '
                            'Prediksi "perlu pemeriksaan lanjutan" diberikan jika probabilitas ≥ 0,50. '
                            f'Ambang sensitif {fmt(youden)} (Youden) menangkap lebih banyak penderita '
                            'dengan konsekuensi lebih banyak alarm palsu.</p>', unsafe_allow_html=True)

        if 'last_input' in st.session_state:
            st.subheader("Posisi pasien dibandingkan data training")
            df = load_data()
            med = df.groupby('Status')[FEATURES].median()
            rows = []
            for feat in FEATURES:
                rows.append({'Pemeriksaan': LABEL[feat],
                             'Pasien': st.session_state['last_input'][feat],
                             'Median Non-Diabetes': med.loc['Non-Diabetes', feat],
                             'Median Diabetes': med.loc['Diabetes', feat]})
            st.dataframe(pd.DataFrame(rows).set_index('Pemeriksaan').round(3), width='stretch')

    with tab_batch:
        st.markdown('<p class="note">Unggah file CSV dengan kolom: ' + ", ".join(FEATURES) +
                    '. Setiap baris dihitung sebagai satu pasien.</p>', unsafe_allow_html=True)
        template = pd.DataFrame([{f: RANGE[f][2] for f in FEATURES}])
        st.download_button("Unduh contoh format CSV", template.to_csv(index=False), "contoh_input.csv", "text/csv")
        uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"], key="diab_csv")
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            missing = [c for c in FEATURES if c not in input_df.columns]
            if missing:
                st.error(f"Kolom berikut tidak ditemukan di file: {', '.join(missing)}. "
                         "Samakan nama kolom dengan contoh format CSV.")
                return
            X = input_df[FEATURES].apply(pd.to_numeric, errors='coerce')
            bad = X.isna().any(axis=1)
            if bad.any():
                st.warning(f"{int(bad.sum())} baris berisi nilai kosong atau bukan angka dan dilewati.")
            X = X[~bad]
            if X.empty:
                st.error("Tidak ada baris yang bisa dihitung.")
                return
            proba = load_diabetes_model().predict_proba(X)[:, 1]
            out = X.copy()
            out['Probabilitas diabetes'] = proba.round(3)
            out['Hasil'] = np.where(proba >= 0.5, 'Perlu pemeriksaan lanjutan', 'Risiko rendah')
            k1, k2 = st.columns(2)
            k1.metric("Pasien dihitung", len(out))
            k2.metric("Perlu pemeriksaan lanjutan", int((proba >= 0.5).sum()))
            st.dataframe(out, width='stretch')
            st.download_button("Unduh hasil", out.to_csv(index=False), "hasil_skrining.csv", "text/csv")


# ---------------------------------------------------------------------------
# Halaman 2: Data pasien (EDA)
# ---------------------------------------------------------------------------
def page_data():
    df = load_data()
    st.title("Data pasien")
    st.markdown('<p class="lede">Pima Indians Diabetes Database setelah pembersihan di Capstone 1: '
                'perempuan keturunan Pima berusia minimal 21 tahun.</p>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Jumlah pasien", f"{len(df)}")
    k2.metric("Penderita diabetes", fmt(df['Outcome'].mean()*100, 1) + "%")
    k3.metric("Median glukosa, diabetes", f"{df.loc[df.Outcome==1,'Glucose'].median():.0f} mg/dl")
    k4.metric("Median glukosa, non-diabetes", f"{df.loc[df.Outcome==0,'Glucose'].median():.0f} mg/dl")

    left, right = st.columns([1.4, 1], gap="large")
    with left:
        st.subheader("Sebaran per pemeriksaan")
        feat = st.selectbox("Pemeriksaan", FEATURES, format_func=lambda f: LABEL[f])
        fig = px.histogram(df, x=feat, color='Status', barmode='overlay', opacity=0.7, nbins=30,
                           color_discrete_map=STATUS_COLORS, labels={feat: LABEL[feat], 'count': 'Jumlah pasien'})
        fig.update_yaxes(title="Jumlah pasien")
        st.plotly_chart(style_fig(fig), width='stretch')
    with right:
        st.subheader("Korelasi dengan diabetes")
        corr = df[FEATURES + ['Outcome']].corr()['Outcome'].drop('Outcome').sort_values()
        fig = go.Figure(go.Bar(x=corr.values, y=[LABEL[i] for i in corr.index], orientation='h',
                               marker_color=[AMBER if v == corr.max() else BLUE for v in corr.values],
                               text=[f"{v:.2f}" for v in corr.values], textposition='outside'))
        fig.update_xaxes(range=[0, max(0.6, corr.max() + 0.1)], title="Korelasi Pearson (r)")
        st.plotly_chart(style_fig(fig), width='stretch')
        st.markdown('<p class="note">Glukosa paling kuat berhubungan dengan diabetes. '
                    'Korelasi menunjukkan hubungan linear, bukan sebab-akibat.</p>', unsafe_allow_html=True)

    with st.expander("Lihat tabel data"):
        st.dataframe(df[FEATURES + ['Status']], width='stretch')


# ---------------------------------------------------------------------------
# Halaman 3: Performa model
# ---------------------------------------------------------------------------
def page_model():
    report = load_report()
    st.title("Performa model")
    if report is None:
        st.info("File model_report.json belum ada. Jalankan sel terakhir Capstone 2, "
                "lalu letakkan file tersebut di folder yang sama dengan app.py.")
        return
    best = report['best_model']
    st.markdown(f'<p class="lede">Empat model dilatih pada {report["n_train"]} pasien dan diuji pada '
                f'{report["n_test"]} pasien yang tidak pernah dilihat model. Model yang dipakai dashboard: '
                f'<strong>{best}</strong>, dipilih berdasarkan ROC-AUC 5-fold cross validation.</p>',
                unsafe_allow_html=True)

    met = pd.DataFrame(report['metrics']).set_index('Model')
    st.subheader("Perbandingan model")
    st.dataframe(met.style.format("{:.3f}").apply(
        lambda r: ['background-color: #E6EEF8; font-weight: 600' if r.name == best else '' for _ in r], axis=1),
        width='stretch')
    st.markdown('<p class="note">Selisih AUC antara tiga model terbaik di bawah 0,02. Dengan data uji sebesar ini, '
                'selisih tersebut belum cukup untuk menyatakan satu model lebih unggul. Perbedaan yang nyata '
                'ada pada recall dan presisi.</p>', unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Kurva ROC")
        fig = go.Figure()
        palette = {'Logistic Regression': '#7A8AA0', 'Random Forest': BLUE, 'Decision Tree': '#B7C2CF', 'MLP': '#4C9A8F'}
        auc = met['Test AUC'].to_dict()
        for name, pts in report['roc'].items():
            fig.add_trace(go.Scatter(x=pts['fpr'], y=pts['tpr'], mode='lines', name=f"{name} ({auc[name]:.2f})",
                                     line=dict(color=palette.get(name, MUTED), width=3.5 if name == best else 1.8)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Tebakan acak (0.50)',
                                 line=dict(color=MUTED, dash='dash', width=1)))
        fig.update_xaxes(title="False positive rate", range=[0, 1])
        fig.update_yaxes(title="True positive rate", range=[0, 1.02])
        style_fig(fig, 400).update_layout(legend=dict(orientation="v", x=0.98, y=0.04, xanchor="right",
                                                      yanchor="bottom", bgcolor="rgba(255,255,255,0.9)",
                                                      bordercolor=LINE, borderwidth=1))
        st.plotly_chart(fig, width='stretch')
    with right:
        st.subheader("Confusion matrix")
        cm = np.array(report['confusion_matrix'])
        fig = go.Figure(go.Heatmap(z=cm, x=['Prediksi Non-Diabetes', 'Prediksi Diabetes'],
                                   y=['Aktual Non-Diabetes', 'Aktual Diabetes'],
                                   colorscale=[[0, '#F2F5F9'], [1, BLUE]], showscale=False,
                                   text=cm, texttemplate="%{text}", textfont=dict(size=22)))
        fig.update_yaxes(autorange='reversed')
        st.plotly_chart(style_fig(fig, 400), width='stretch')
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f'<p class="note">Model {best} pada threshold 0,50. Dari {tp+fn} penderita diabetes di data uji, {tp} terdeteksi dan {fn} terlewat. '
                    f'Dari {tn+fp} pasien non-diabetes, {fp} mendapat alarm palsu.</p>', unsafe_allow_html=True)

    if report.get('feature_importance'):
        st.subheader("Pemeriksaan yang paling berpengaruh pada model")
        imp = pd.Series(report['feature_importance']).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=[LABEL[i] for i in imp.index], orientation='h',
                               marker_color=BLUE, text=[f"{v:.2f}" for v in imp.values], textposition='outside'))
        fig.update_xaxes(title="Feature importance" if report['importance_type'] == 'feature_importances_' else "Koefisien")
        st.plotly_chart(style_fig(fig, 320), width='stretch')


# ---------------------------------------------------------------------------
# Halaman 4: Iris species (bagian dari ketentuan template)
# ---------------------------------------------------------------------------
IRIS_COLS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
IRIS_NAMES = ['Iris setosa', 'Iris versicolor', 'Iris virginica']


@st.cache_resource
def load_iris_model():
    if os.path.exists("generate_iris.pkl"):
        with open("generate_iris.pkl", "rb") as file:
            return pickle.load(file)
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    iris_data = load_iris(as_frame=True)
    X = iris_data.data.copy()
    X.columns = IRIS_COLS
    return RandomForestClassifier(random_state=42).fit(X, iris_data.target)


def page_iris():
    st.title("Prediksi spesies Iris")
    st.markdown('<p class="lede">Contoh tambahan sesuai ketentuan tugas: memprediksi spesies bunga Iris '
                'dari ukuran kelopak dan mahkota. Data: Iris dataset (UCI ML).</p>', unsafe_allow_html=True)
    left, right = st.columns([1, 1], gap="large")
    with left:
        uploaded_file = st.file_uploader("Unggah CSV (opsional)", type=["csv"], key="iris_csv",
                                         help="Kolom: " + ", ".join(IRIS_COLS))
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
        else:
            input_df = pd.DataFrame([{
                'sepal_length': st.slider('Panjang sepal (cm)', 4.3, 7.9, 5.8),
                'sepal_width': st.slider('Lebar sepal (cm)', 2.0, 4.4, 3.0),
                'petal_length': st.slider('Panjang petal (cm)', 1.0, 6.9, 4.3),
                'petal_width': st.slider('Lebar petal (cm)', 0.1, 2.5, 1.3)}])
        go_predict = st.button("Prediksi spesies", type="primary")
    with right:
        if go_predict:
            missing = [c for c in IRIS_COLS if c not in input_df.columns]
            if missing:
                st.error(f"Kolom berikut tidak ditemukan di file: {', '.join(missing)}.")
                return
            df = input_df[IRIS_COLS]
            pred = load_iris_model().predict(df)
            if len(df) == 1:
                st.success(f"Prediksi spesies: {IRIS_NAMES[int(pred[0])]}")
                st.dataframe(df, width='stretch')
            else:
                out = df.copy()
                out['Prediksi'] = [IRIS_NAMES[int(p)] for p in pred]
                st.dataframe(out, width='stretch')
        else:
            st.markdown('<div class="empty">Atur ukuran bunga atau unggah CSV, lalu tekan <em>Prediksi spesies</em>.</div>',
                        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Navigasi
# ---------------------------------------------------------------------------
def main():
    with st.sidebar:
        st.markdown("### Skrining Diabetes")
        st.markdown('<p class="note">Capstone Project: model klasifikasi diabetes berbasis data klinis.</p>',
                    unsafe_allow_html=True)
        add_selectitem = st.radio("Halaman", ["Skrining pasien", "Data pasien", "Performa model", "Iris species"],
                                  label_visibility="collapsed")
        st.divider()
        st.markdown('<p class="note">Dibuat oleh: [ISI NAMA KAMU]<br>'
                    'Data: <a href="https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database">'
                    'Pima Indians Diabetes Database (Kaggle)</a></p>', unsafe_allow_html=True)
        st.markdown('<p class="note">Alat bantu skrining, bukan pengganti diagnosis tenaga kesehatan.</p>',
                    unsafe_allow_html=True)

    if add_selectitem == "Skrining pasien":
        page_screening()
    elif add_selectitem == "Data pasien":
        page_data()
    elif add_selectitem == "Performa model":
        page_model()
    else:
        page_iris()


if __name__ == "__main__":
    main()
