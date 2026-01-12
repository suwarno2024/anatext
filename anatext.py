import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from openai import OpenAI
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="AnaText - AI Text Analysis",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STATE MANAGEMENT ---
if 'stop_words' not in st.session_state:
    # Default Stop Words (Campuran Indo & Inggris dasar)
    st.session_state.stop_words = [
        "yang", "di", "dan", "itu", "dengan", "untuk", "tidak", "ini", "dari", 
        "dalam", "akan", "pada", "juga", "saya", "adalah", "ke", "karena", 
        "bisa", "ada", "mereka", "kita", "kamu", "the", "and", "is", "of", "to", "in"
    ]
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# --- CSS CUSTOM (UI MODERN) ---
def inject_custom_css(mode):
    if mode == 'Dark':
        bg_color = "#0e1117"
        text_color = "#fafafa"
        card_bg = "#262730"
        border_col = "#4c4e56"
    else:
        bg_color = "#ffffff"
        text_color = "#31333F"
        card_bg = "#f0f2f6"
        border_col = "#d1d5db"

    st.markdown(f"""
    <style>
        .stApp {{ background-color: {bg_color}; color: {text_color}; }}
        
        /* Drag & Drop Area Styling */
        [data-testid='stFileUploader'] {{
            background-color: {card_bg};
            border: 2px dashed #4c7bf4;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        
        /* Button Styling */
        .stButton button {{
            border-radius: 8px;
            font-weight: bold;
            width: 100%;
        }}
        
        /* Custom highlight for sentiment table */
        .positive-bg {{background-color: #d1fae5; color: #065f46; padding: 4px 8px; border-radius: 4px;}}
        .negative-bg {{background-color: #fee2e2; color: #991b1b; padding: 4px 8px; border-radius: 4px;}}
        .neutral-bg {{background-color: #f3f4f6; color: #1f2937; padding: 4px 8px; border-radius: 4px;}}
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOGIKA UTAMA ---

def clean_text(text, remove_sw, use_lemma, case_folding, stopwords_list, stemmer):
    """Membersihkan teks sesuai konfigurasi user."""
    if not isinstance(text, str):
        return ""
    
    # 1. Case Folding
    if case_folding:
        text = text.lower()
    
    # Bersihkan karakter non-alphanumeric
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    
    # 2. Hapus Stop Words
    if remove_sw:
        tokens = [word for word in tokens if word not in stopwords_list]
    
    # 3. Lemmatization (Sastrawi)
    if use_lemma and stemmer:
        # Loop manual demi akurasi (meski agak lambat)
        tokens = [stemmer.stem(word) for word in tokens]
    
    return " ".join(tokens)

def get_sentiment_ai(client, model, text_list):
    """Analisis sentimen menggunakan OpenAI."""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(text_list)
    
    for i, text in enumerate(text_list):
        if not text.strip():
            results.append("Netral")
            continue 
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Klasifikasikan sentimen: Positif, Negatif, atau Netral. Jawab 1 kata saja."},
                    {"role": "user", "content": text}
                ],
                temperature=0,
                max_tokens=10
            )
            sentiment = response.choices[0].message.content.strip().replace(".", "")
            results.append(sentiment)
        except Exception:
            results.append("Error")
        
        # Update progress visual
        if i % 5 == 0 or i == total - 1: # Update tiap 5 item agar UI responsif
            progress_bar.progress((i + 1) / total)
            status_text.text(f"Menganalisis sentimen... {i+1}/{total}")
    
    progress_bar.empty()
    status_text.empty()
    return results

def get_topic_name_ai(client, model, keywords):
    """Memberi nama topik berdasarkan keyword."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Berikan nama topik singkat (2-4 kata) dari keyword ini."},
                {"role": "user", "content": f"Keywords: {', '.join(keywords)}"}
            ]
        )
        return response.choices[0].message.content.replace('"', '')
    except:
        return "Topik Umum"

# --- MODAL/DIALOG HANDLER (SAFE MODE) ---
# Bagian ini menangani error 'AttributeError' dengan mengecek ketersediaan fitur

def show_stopwords_manager():
    """Fungsi konten untuk mengelola stopwords."""
    st.write("Tambahkan atau hapus kata yang tidak ingin dianalisis.")
    
    # Input kata baru
    col_in1, col_in2 = st.columns([3, 1])
    with col_in1:
        new_word = st.text_input("Tambah kata baru:", label_visibility="collapsed", placeholder="Ketik kata...")
    with col_in2:
        add_btn = st.button("Tambah")
    
    if add_btn and new_word:
        if new_word.lower() not in st.session_state.stop_words:
            st.session_state.stop_words.append(new_word.lower())
            st.rerun()

    # Tampilan Multiselect (sebagai Tags)
    current_words = st.multiselect(
        "Daftar Stop Words:",
        options=st.session_state.stop_words,
        default=st.session_state.stop_words
    )
    
    # Sinkronisasi jika ada yang dihapus via tanda 'x'
    if len(current_words) != len(st.session_state.stop_words):
        st.session_state.stop_words = current_words
        st.rerun()

# Logika Pemilihan Tampilan (Dialog vs Expander)
if hasattr(st, "dialog"):
    # Jika Streamlit versi baru (Support Dialog)
    @st.dialog("Kelola Stop Words")
    def open_stopwords_modal():
        show_stopwords_manager()
        if st.button("Simpan & Tutup", type="primary"):
            st.rerun()
else:
    # Jika Streamlit versi lama (Fallback ke Expander)
    def open_stopwords_modal():
        # Karena expander tidak bisa dipanggil sebagai fungsi event, 
        # kita handle logika tampilannya langsung di Sidebar nanti.
        pass 


# --- SIDEBAR (PENGATURAN) ---
with st.sidebar:
    st.title("⚙️ Pengaturan")
    
    # Theme Toggle
    theme_mode = st.radio("Tema Tampilan", ["Light", "Dark"], horizontal=True)
    inject_custom_css(theme_mode)
    
    st.divider()
    
    st.subheader("Bahasa & Teks")
    language = st.selectbox("Bahasa Teks", ["Indonesia", "Inggris"])
    text_type = st.selectbox("Tipe Teks", ["Umum", "Ulasan Produk", "Media Sosial", "Berita"])
    
    st.divider()
    
    st.subheader("🤖 Model AI")
    granularity = st.selectbox("Granularitas Sentimen", ["Dasar (Positif/Netral/Negatif)", "Lanjut"])
    num_clusters_input = st.slider("Jumlah Topik (Klaster)", 2, 10, 5)
    
    st.divider()
    
    st.subheader("🔧 Preprocessing")
    check_sw = st.checkbox("Hapus Stop Words", value=True)
    check_lemma = st.checkbox("Aktifkan Lemmatization", value=True)
    check_lower = st.checkbox("Case Folding (lowercase)", value=True)
    
    # Tombol Kelola Stopwords
    if hasattr(st, "dialog"):
        if st.button("Kelola Stop Words", use_container_width=True):
            open_stopwords_modal()
    else:
        # Fallback UI untuk versi lama
        with st.expander("Kelola Stop Words"):
            show_stopwords_manager()

# --- MAIN PAGE ---
col_logo, col_title = st.columns([1, 10])
with col_title:
    st.title("AnaText")
    st.write("Platform Analisis Teks Berbasis AI")

# Setup API Key (Silent)
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    api_key = "" 

client = OpenAI(api_key=api_key) if api_key else None
MODEL_NAME = "gpt-4o" 

# --- INPUT AREA ---
container_input = st.container()
with container_input:
    tab_upload, tab_text = st.tabs(["📂 Unggah Dokumen", "✍️ Teks Langsung"])
    input_text_list = []
    
    with tab_upload:
        st.info("Mendukung format .txt, .csv, .xlsx")
        uploaded_file = st.file_uploader("Klik atau Seret File ke Sini", type=['csv', 'xlsx', 'txt'])
        
        if uploaded_file:
            # FIX: Encoding Handling (UTF-8 vs Latin-1)
            if uploaded_file.name.endswith('.csv'):
                try:
                    df_upload = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    df_upload = pd.read_csv(uploaded_file, encoding='latin-1')
            elif uploaded_file.name.endswith('.xlsx'):
                df_upload = pd.read_excel(uploaded_file)
            else:
                bytes_data = uploaded_file.read()
                try:
                    raw_text = bytes_data.decode("utf-8")
                except UnicodeDecodeError:
                    raw_text = bytes_data.decode("latin-1")
                df_upload = pd.DataFrame(raw_text.splitlines(), columns=['Teks'])
            
            # Auto detect text column
            possible_cols = [c for c in df_upload.columns if df_upload[c].dtype == 'object']
            if possible_cols:
                text_col = st.selectbox("Konfirmasi Kolom Teks:", possible_cols)
                input_text_list = df_upload[text_col].dropna().astype(str).tolist()
            else:
                st.error("File tidak memiliki kolom teks yang valid.")

    with tab_text:
        direct_text = st.text_area("Tempelkan teks di sini...", height=150)
        if direct_text:
            input_text_list = [t for t in direct_text.split('\n') if t.strip()]

# --- PROSES ANALISIS ---
if st.button("🚀 Lakukan Analisis", type="primary"):
    if not input_text_list:
        st.warning("Mohon masukkan data teks terlebih dahulu.")
    elif not client:
        st.error("API Key belum terpasang. Cek Secrets Anda.")
    else:
        with st.spinner("Sedang memproses..."):
            # 1. Load Data
            df = pd.DataFrame(input_text_list, columns=['Teks_Asli'])
            
            # 2. Preprocessing Loop
            factory = StemmerFactory()
            stemmer = factory.create_stemmer() if (language == "Indonesia" and check_lemma) else None
            
            clean_results = []
            progress_bar = st.progress(0)
            
            for idx, text in enumerate(df['Teks_Asli']):
                cleaned = clean_text(text, check_sw, check_lemma, check_lower, st.session_state.stop_words, stemmer)
                clean_results.append(cleaned)
                if idx % 10 == 0: progress_bar.progress((idx+1)/len(df))
            
            df['Teks_Clean'] = clean_results
            progress_bar.empty()

            # Filter data kosong setelah cleaning
            df = df[df['Teks_Clean'].str.strip() != ""]

            # 3. TF-IDF & Clustering
            # FIX: Handle jika data < jumlah klaster
            actual_clusters = min(num_clusters_input, len(df))
            if actual_clusters < 2: actual_clusters = 1 # Fallback minimal

            vectorizer = TfidfVectorizer(max_features=2000)
            tfidf_matrix = vectorizer.fit_transform(df['Teks_Clean'])
            feature_names = vectorizer.get_feature_names_out()
            
            kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
            kmeans.fit(tfidf_matrix)
            df['Cluster_ID'] = kmeans.labels_

            # 4. Labeling Topik (AI)
            cluster_names = {}
            for i in range(actual_clusters):
                centroid = kmeans.cluster_centers_[i]
                top_indices = centroid.argsort()[-5:][::-1]
                top_words = [feature_names[ind] for ind in top_indices]
                
                if top_words:
                    label = get_topic_name_ai(client, MODEL_NAME, top_words)
                    cluster_names[i] = label
                else:
                    cluster_names[i] = f"Topik {i+1}"
            
            df['Topik'] = df['Cluster_ID'].map(cluster_names)

            # 5. Sentimen (AI) - Full Data
            df['Sentimen'] = get_sentiment_ai(client, MODEL_NAME, df['Teks_Asli'].tolist())

            # Save to Session
            st.session_state.data = df
            st.session_state.vectorizer = vectorizer
            st.session_state.tfidf_matrix = tfidf_matrix
            st.session_state.analysis_done = True
            st.rerun()

# --- DASHBOARD HASIL ---
if st.session_state.analysis_done and st.session_state.data is not None:
    df = st.session_state.data
    st.write("---")
    st.subheader("📊 Insight Dashboard")
    
    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Dokumen", len(df))
    sentiment_counts = df['Sentimen'].value_counts()
    top_sent = sentiment_counts.idxmax() if not sentiment_counts.empty else "-"
    m2.metric("Dominasi Sentimen", top_sent)
    m3.metric("Jumlah Topik", df['Topik'].nunique())

    # Tabs Visualisasi
    tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan", "Sentimen", "Topik", "Kata Kunci"])

    with tab1:
        st.info("Ringkasan dihasilkan oleh AI berdasarkan statistik data.")
        if st.button("Generate Executive Summary"):
            with st.spinner("AI sedang menulis..."):
                try:
                    prompt = f"Data: {len(df)} baris. Sentimen: {sentiment_counts.to_dict()}. Topik: {df['Topik'].value_counts().head(3).index.tolist()}. Buat ringkasan eksekutif singkat padat."
                    res = client.chat.completions.create(
                        model=MODEL_NAME, messages=[{"role":"user", "content": prompt}]
                    )
                    st.markdown(res.choices[0].message.content)
                except: st.error("Gagal koneksi AI.")

    with tab2:
        c1, c2 = st.columns([1, 2])
        with c1:
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, hole=0.5, 
                         color=sentiment_counts.index, 
                         color_discrete_map={'Positif':'#34d399', 'Negatif':'#f87171', 'Netral':'#9ca3af'})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.write("Detail Data:")
            filter_s = st.multiselect("Filter:", df['Sentimen'].unique(), default=df['Sentimen'].unique())
            
            def color_sent(val):
                if val=='Positif': return 'background-color: #d1fae5; color: #065f46'
                if val=='Negatif': return 'background-color: #fee2e2; color: #991b1b'
                return 'background-color: #f3f4f6'
            
            st.dataframe(df[df['Sentimen'].isin(filter_s)][['Teks_Asli','Sentimen']].style.map(color_sent, subset=['Sentimen']), use_container_width=True, height=400)

    with tab3:
        topic_counts = df['Topik'].value_counts().reset_index()
        topic_counts.columns = ['Topik', 'Jumlah']
        fig_bar = px.bar(topic_counts, x='Jumlah', y='Topik', orientation='h', color='Jumlah')
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab4:
        # WordCloud
        text_all = " ".join(df['Teks_Clean'])
        if text_all.strip():
            wc = WordCloud(width=800, height=400, background_color='white').generate(text_all)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear'); plt.axis("off")
            st.pyplot(plt)
        else:
            st.warning("Tidak ada kata kunci yang cukup untuk Word Cloud.")
        
        # TF-IDF Table
        st.write("**Top Kata Unik (TF-IDF)**")
        sum_tfidf = st.session_state.tfidf_matrix.sum(axis=0)
        words = [(word, sum_tfidf[0, idx]) for word, idx in st.session_state.vectorizer.vocabulary_.items()]
        words = sorted(words, key=lambda x: x[1], reverse=True)[:10]
        st.table(pd.DataFrame(words, columns=["Kata", "Skor"]))

    # Download
    st.divider()
    st.download_button("📥 Unduh CSV", df.to_csv(index=False).encode('utf-8'), "analisis_anatext.csv", "text/csv")
