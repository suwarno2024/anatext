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

# --- DEFINISI STOPWORDS & COLORS ---
default_stopwords_id = [
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk', 'pada', 'adalah', 'sebagai', 
    'dalam', 'tidak', 'akan', 'juga', 'atau', 'ada', 'mereka', 'sudah', 'saya', 'kita', 'kami', 'kalian', 
    'dia', 'ia', 'anda', 'bisa', 'hanya', 'lebih', 'karena', 'tetapi', 'tapi', 'namun', 'jika', 'maka', 
    'oleh', 'saat', 'agar', 'seperti', 'bahwa', 'telah', 'dapat', 'menjadi', 'tersebut', 'sangat', 'sehingga', 
    'secara', 'antara', 'sebuah', 'suatu', 'begitu', 'lagi', 'masih', 'banyak', 'semua', 'setiap', 'serta', 
    'hal', 'bila', 'pun', 'lalu', 'kemudian', 'yakni', 'yaitu', 'apabila', 'ketika', 'baik', 'paling', 
    'demi', 'hingga', 'sampai', 'tanpa', 'belum', 'harus', 'sedang', 'maupun', 'selain', 'melalui', 
    'sendiri', 'beberapa', 'apa', 'siapa', 'mana', 'kapan', 'bagaimana', 'mengapa', 'kenapa'
]

default_stopwords_en = [
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 
    'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 
    'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 
    'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 
    'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 
    'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 
    'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 
    'give', 'day', 'most', 'us', 'is', 'are', 'was', 'were', 'been', 'being', 'has', 'had', 'having', 
    'does', 'did', 'doing', 'am'
]

# Warna Sentimen (Standardized)
SENTIMENT_COLORS = {
    'Positif': '#28a745',  # Hijau
    'Negatif': '#dc3545',  # Merah
    'Netral': '#ffc107',   # Kuning (Warning/Gold)
    'Error': '#6c757d'
}

# --- STATE MANAGEMENT ---
if 'stop_words' not in st.session_state:
    # Gabungkan kedua list stopwords saat inisialisasi
    st.session_state.stop_words = list(set(default_stopwords_id + default_stopwords_en))

if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# --- CSS CUSTOM (THEME & CONTRAST) ---
def inject_custom_css(mode):
    # Logika Kontras Tinggi
    if mode == 'Dark':
        bg_color = "#0e1117"
        text_color = "#ffffff" # Putih absolut untuk kontras
        card_bg = "#262730"
        border_col = "#4c4e56"
        metric_val_col = "#4cdbc4" # Warna angka metric cerah
    else:
        bg_color = "#ffffff"
        text_color = "#000000" # Hitam absolut untuk kontras
        card_bg = "#f0f2f6"
        border_col = "#d1d5db"
        metric_val_col = "#000000"

    st.markdown(f"""
    <style>
        .stApp {{ background-color: {bg_color}; color: {text_color}; }}
        
        /* Global Text Contrast */
        p, h1, h2, h3, h4, h5, h6, li, span {{
            color: {text_color} !important;
        }}
        
        /* Drag & Drop Area */
        [data-testid='stFileUploader'] {{
            background-color: {card_bg};
            border: 2px dashed #4c7bf4;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        
        /* Metric Styling */
        [data-testid="stMetricValue"] {{
            color: {metric_val_col} !important;
        }}
        
        /* Highlight Sentimen Tabel - Text tetap gelap agar terbaca di background warna */
        .positive-bg {{background-color: #28a745; color: #ffffff; padding: 4px 8px; border-radius: 4px; font-weight: bold;}}
        .negative-bg {{background-color: #dc3545; color: #ffffff; padding: 4px 8px; border-radius: 4px; font-weight: bold;}}
        .neutral-bg {{background-color: #ffc107; color: #000000; padding: 4px 8px; border-radius: 4px; font-weight: bold;}}
        
        /* Buttons */
        .stButton button {{
            font-weight: bold;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    return "plotly_dark" if mode == 'Dark' else "plotly_white"

# --- FUNGSI LOGIKA ---

def clean_text(text, remove_sw, use_lemma, case_folding, stopwords_list, stemmer):
    if not isinstance(text, str):
        return ""
    
    if case_folding:
        text = text.lower()
    
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    
    if remove_sw:
        tokens = [word for word in tokens if word not in stopwords_list]
    
    if use_lemma and stemmer:
        tokens = [stemmer.stem(word) for word in tokens]
    
    return " ".join(tokens)

def get_sentiment_ai(client, model, text_list):
    results = []
    progress_bar = st.progress(0)
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
                temperature=0, max_tokens=10
            )
            sentiment = response.choices[0].message.content.strip().replace(".", "")
            # Normalisasi output AI agar sesuai key warna
            if "positif" in sentiment.lower(): sentiment = "Positif"
            elif "negatif" in sentiment.lower(): sentiment = "Negatif"
            else: sentiment = "Netral"
            
            results.append(sentiment)
        except Exception:
            results.append("Error")
        
        if i % 5 == 0 or i == total - 1:
            progress_bar.progress((i + 1) / total)
    
    progress_bar.empty()
    return results

def get_topic_name_ai(client, model, keywords):
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

# --- MODAL MANAGER (SAFE MODE) ---
def show_stopwords_manager():
    col_in1, col_in2 = st.columns([3, 1])
    with col_in1:
        new_word = st.text_input("Tambah kata baru:", label_visibility="collapsed", placeholder="Ketik kata...")
    with col_in2:
        if st.button("Tambah"):
            if new_word and new_word.lower() not in st.session_state.stop_words:
                st.session_state.stop_words.append(new_word.lower())
                st.rerun()

    current_words = st.multiselect(
        "Daftar Stop Words:",
        options=st.session_state.stop_words,
        default=st.session_state.stop_words
    )
    
    if len(current_words) != len(st.session_state.stop_words):
        st.session_state.stop_words = current_words
        st.rerun()

if hasattr(st, "dialog"):
    @st.dialog("Kelola Stop Words")
    def open_stopwords_modal():
        show_stopwords_manager()
        if st.button("Simpan & Tutup", type="primary"):
            st.rerun()
else:
    def open_stopwords_modal(): pass

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Pengaturan")
    
    # Theme Mode dengan Logika Grafik
    theme_mode = st.radio("Tema Tampilan", ["Light", "Dark"], horizontal=True)
    plotly_template = inject_custom_css(theme_mode) # Return template grafik
    
    st.divider()
    
    st.subheader("Bahasa & Teks")
    language = st.selectbox("Bahasa Teks", ["Indonesia", "Inggris"])
    text_type = st.selectbox("Tipe Teks", ["Umum", "Ulasan Produk", "Media Sosial", "Berita"])
    
    st.divider()
    st.subheader("🤖 Model AI")
    num_clusters_input = st.slider("Jumlah Topik (Klaster)", 2, 10, 5)
    
    st.divider()
    st.subheader("🔧 Preprocessing")
    check_sw = st.checkbox("Hapus Stop Words", value=True)
    check_lemma = st.checkbox("Aktifkan Lemmatization", value=True)
    check_lower = st.checkbox("Case Folding (lowercase)", value=True)
    
    if hasattr(st, "dialog"):
        if st.button("Kelola Stop Words", use_container_width=True):
            open_stopwords_modal()
    else:
        with st.expander("Kelola Stop Words"):
            show_stopwords_manager()

# --- HALAMAN UTAMA ---
col_logo, col_title = st.columns([1, 10])
with col_title:
    st.title("AnaText")
    st.write("Platform Analisis Teks Berbasis AI")

# API Check
try: api_key = st.secrets["OPENAI_API_KEY"]
except: api_key = "" 
client = OpenAI(api_key=api_key) if api_key else None
MODEL_NAME = "gpt-4o" 

# --- INPUT AREA ---
container_input = st.container()
with container_input:
    tab_upload, tab_text = st.tabs(["📂 Unggah Dokumen", "✍️ Teks Langsung"])
    input_text_list = []
    
    with tab_upload:
        st.info("Limit file: 10 MB. Mendukung .txt, .csv, .xlsx")
        uploaded_file = st.file_uploader("Klik atau Seret File ke Sini", type=['csv', 'xlsx', 'txt'])
        
        if uploaded_file:
            # 1. CEK LIMIT 10MB (10 * 1024 * 1024 bytes)
            if uploaded_file.size > 10 * 1024 * 1024:
                st.error("⚠️ Ukuran file melebihi batas 10 MB. Harap unggah file yang lebih kecil.")
                input_text_list = [] # Reset
            else:
                # Proses File jika ukuran aman
                if uploaded_file.name.endswith('.csv'):
                    try: df_upload = pd.read_csv(uploaded_file, encoding='utf-8')
                    except: df_upload = pd.read_csv(uploaded_file, encoding='latin-1')
                elif uploaded_file.name.endswith('.xlsx'):
                    df_upload = pd.read_excel(uploaded_file)
                else:
                    bytes_data = uploaded_file.read()
                    try: raw_text = bytes_data.decode("utf-8")
                    except: raw_text = bytes_data.decode("latin-1")
                    df_upload = pd.DataFrame(raw_text.splitlines(), columns=['Teks'])
                
                possible_cols = [c for c in df_upload.columns if df_upload[c].dtype == 'object']
                if possible_cols:
                    text_col = st.selectbox("Konfirmasi Kolom Teks:", possible_cols)
                    input_text_list = df_upload[text_col].dropna().astype(str).tolist()
                else:
                    st.error("File tidak memiliki kolom teks valid.")

    with tab_text:
        direct_text = st.text_area("Tempelkan teks...", height=150)
        if direct_text:
            input_text_list = [t for t in direct_text.split('\n') if t.strip()]

# --- PROSES ---
if st.button("🚀 Lakukan Analisis", type="primary"):
    if not input_text_list:
        st.warning("Masukkan data teks.")
    elif not client:
        st.error("API Key error.")
    else:
        with st.spinner("Processing..."):
            df = pd.DataFrame(input_text_list, columns=['Teks_Asli'])
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
            df = df[df['Teks_Clean'].str.strip() != ""]

            # Clustering
            actual_clusters = min(num_clusters_input, len(df))
            if actual_clusters < 2: actual_clusters = 1
            
            vectorizer = TfidfVectorizer(max_features=2000)
            tfidf_matrix = vectorizer.fit_transform(df['Teks_Clean'])
            feature_names = vectorizer.get_feature_names_out()
            
            kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
            kmeans.fit(tfidf_matrix)
            df['Cluster_ID'] = kmeans.labels_

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
            df['Sentimen'] = get_sentiment_ai(client, MODEL_NAME, df['Teks_Asli'].tolist())

            st.session_state.data = df
            st.session_state.vectorizer = vectorizer
            st.session_state.tfidf_matrix = tfidf_matrix
            st.session_state.analysis_done = True
            st.rerun()

# --- DASHBOARD ---
if st.session_state.analysis_done and st.session_state.data is not None:
    df = st.session_state.data
    st.write("---")
    st.subheader("📊 Insight Dashboard")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Dokumen", len(df))
    sentiment_counts = df['Sentimen'].value_counts()
    top_sent = sentiment_counts.idxmax() if not sentiment_counts.empty else "-"
    m2.metric("Dominasi Sentimen", top_sent)
    m3.metric("Jumlah Topik", df['Topik'].nunique())

    tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan", "Sentimen", "Topik", "Kata Kunci"])

    with tab1:
        if st.button("Generate Summary"):
            with st.spinner("AI writing..."):
                try:
                    prompt = f"Data: {len(df)} baris. Sentimen: {sentiment_counts.to_dict()}. Topik: {df['Topik'].value_counts().head(3).index.tolist()}. Buat ringkasan singkat."
                    res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role":"user", "content": prompt}])
                    st.markdown(res.choices[0].message.content)
                except: st.error("AI Busy.")

    with tab2:
        c1, c2 = st.columns([1, 2])
        with c1:
            # PIE CHART DENGAN WARNA TETAP (HIJAU, MERAH, KUNING)
            fig = px.pie(
                values=sentiment_counts.values, 
                names=sentiment_counts.index, 
                hole=0.5, 
                color=sentiment_counts.index, 
                color_discrete_map=SENTIMENT_COLORS,
                template=plotly_template # Kontras Light/Dark
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.write("Detail Data:")
            filter_s = st.multiselect("Filter:", df['Sentimen'].unique(), default=df['Sentimen'].unique())
            
            def color_sent(val):
                if val == 'Positif': return 'background-color: #28a745; color: #ffffff'
                if val == 'Negatif': return 'background-color: #dc3545; color: #ffffff'
                if val == 'Netral': return 'background-color: #ffc107; color: #000000'
                return ''
            
            st.dataframe(df[df['Sentimen'].isin(filter_s)][['Teks_Asli','Sentimen']].style.map(color_sent, subset=['Sentimen']), use_container_width=True, height=400)

    with tab3:
        topic_counts = df['Topik'].value_counts().reset_index()
        topic_counts.columns = ['Topik', 'Jumlah']
        # BAR CHART KONTRAS
        fig_bar = px.bar(
            topic_counts, x='Jumlah', y='Topik', orientation='h', color='Jumlah',
            template=plotly_template # Kontras Light/Dark
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab4:
        text_all = " ".join(df['Teks_Clean'])
        if text_all.strip():
            wc = WordCloud(width=800, height=400, background_color='black' if theme_mode=='Dark' else 'white').generate(text_all)
            plt.figure(figsize=(10, 5), facecolor='k' if theme_mode=='Dark' else 'w')
            plt.imshow(wc, interpolation='bilinear'); plt.axis("off")
            st.pyplot(plt)
        
        sum_tfidf = st.session_state.tfidf_matrix.sum(axis=0)
        words = [(word, sum_tfidf[0, idx]) for word, idx in st.session_state.vectorizer.vocabulary_.items()]
        words = sorted(words, key=lambda x: x[1], reverse=True)[:10]
        
        # Grafik Bar untuk TF-IDF
        df_tfidf = pd.DataFrame(words, columns=["Kata", "Skor"])
        fig_tfidf = px.bar(df_tfidf, x="Skor", y="Kata", orientation='h', template=plotly_template)
        fig_tfidf.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_tfidf, use_container_width=True)

    st.divider()
    st.download_button("📥 Unduh CSV", df.to_csv(index=False).encode('utf-8'), "analisis_anatext.csv", "text/csv")
