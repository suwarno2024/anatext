import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer # Ditambah CountVectorizer untuk N-Gram
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from openai import OpenAI
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import json # Ditambah untuk parsing output NER

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="AnaText - AI Text Analysis",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STATE MANAGEMENT ---
if 'stop_words' not in st.session_state:
    st.session_state.stop_words = [
        "yang", "di", "dan", "itu", "dengan", "untuk", "tidak", "ini", "dari", 
        "dalam", "akan", "pada", "juga", "saya", "adalah", "ke", "karena", 
        "bisa", "ada", "mereka", "kita", "kamu", "the", "and", "is", "of", "to", "in"
    ]
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'ner_data' not in st.session_state:
    st.session_state.ner_data = None

# --- CSS CUSTOM ---
def inject_custom_css(mode):
    if mode == 'Dark':
        bg_color = "#0e1117"
        text_color = "#fafafa"
        card_bg = "#262730"
    else:
        bg_color = "#ffffff"
        text_color = "#31333F"
        card_bg = "#f0f2f6"

    st.markdown(f"""
    <style>
        .stApp {{ background-color: {bg_color}; color: {text_color}; }}
        h1, h2, h3 {{ font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }}
        [data-testid='stFileUploader'] {{ background-color: {card_bg}; border: 2px dashed #4c7bf4; border-radius: 10px; padding: 20px; text-align: center; }}
        [data-testid='stFileUploader'] section {{ padding: 0; }}
        .stButton button {{ background-color: #2563eb; color: white; border-radius: 8px; padding: 10px 20px; border: none; font-weight: bold; width: 100%; }}
        .stButton button:hover {{ background-color: #1d4ed8; color: white; }}
        .streamlit-expanderHeader {{ background-color: {card_bg}; border-radius: 5px; }}
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI PREPROCESSING ---
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

# --- FUNGSI AI (SENTIMEN, TOPIK, NER) ---
def get_sentiment_ai(client, model, text_list):
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
                temperature=0, max_tokens=10
            )
            sentiment = response.choices[0].message.content.strip().replace(".", "")
            results.append(sentiment)
        except:
            results.append("Error")
        progress_bar.progress((i + 1) / total)
        status_text.text(f"Analisis Sentimen... {i+1}/{total}")
    
    progress_bar.empty()
    status_text.empty()
    return results

def get_topic_name_ai(client, model, keywords):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Berikan nama topik singkat (2-4 kata) berdasarkan kata kunci ini."},
                {"role": "user", "content": f"Keywords: {', '.join(keywords)}"}
            ]
        )
        return response.choices[0].message.content.replace('"', '')
    except:
        return "Topik Tak Teridentifikasi"

def get_ner_ai(client, model, text_list):
    """
    Fungsi untuk ekstraksi Named Entity Recognition menggunakan GPT-4o.
    Output diharapkan berupa JSON list.
    """
    ner_results = {"Person": [], "Organization": [], "Location": []}
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(text_list)

    # Batching sederhaana: Kita proses per baris untuk akurasi maksimal
    # System Prompt yang ketat agar output valid JSON
    system_prompt = """
    Anda adalah ahli NLP (Natural Language Processing). Tugas Anda adalah mengekstrak entitas bernama (NER) dari teks yang diberikan.
    Kategorikan menjadi 3:
    1. PER (Person/Orang)
    2. ORG (Organization/Instansi/Perusahaan)
    3. LOC (Location/Lokasi/Negara/Kota)
    
    KEMBALIKAN HANYA FORMAT JSON VALID SEPERTI INI (TANPA MARKDOWN):
    {
        "PER": ["Nama1", "Nama2"],
        "ORG": ["Org1"],
        "LOC": ["Loc1"]
    }
    Jika tidak ada entitas, kembalikan list kosong [].
    """

    for i, text in enumerate(text_list):
        if not text.strip():
            continue
            
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0,
                response_format={"type": "json_object"} # Memaksa output JSON
            )
            
            # Parsing JSON
            content = response.choices[0].message.content
            data = json.loads(content)
            
            # Agregasi hasil
            if "PER" in data: ner_results["Person"].extend(data["PER"])
            if "ORG" in data: ner_results["Organization"].extend(data["ORG"])
            if "LOC" in data: ner_results["Location"].extend(data["LOC"])
            
        except Exception as e:
            # Skip jika error parsing atau API error
            continue

        progress_bar.progress((i + 1) / total)
        status_text.text(f"Mendeteksi Entitas (NER)... {i+1}/{total}")
    
    progress_bar.empty()
    status_text.empty()
    return ner_results

# --- FUNGSI N-GRAM ---
def get_ngrams(text_series, n=2, top_k=10):
    """
    Menghitung frekuensi N-Gram (Bigram/Trigram) menggunakan CountVectorizer
    """
    try:
        vec = CountVectorizer(ngram_range=(n, n)).fit(text_series)
        bag_of_words = vec.transform(text_series)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:top_k]
    except ValueError:
        return [] # Return kosong jika data terlalu sedikit untuk n-gram

# --- MODAL STOPWORDS ---
@st.experimental_dialog("Kelola Stop Words")
def manage_stopwords_dialog():
    st.write("Tambahkan atau hapus kata yang tidak ingin dianalisis.")
    new_word = st.text_input("Tambah kata baru (tekan Enter):")
    if new_word:
        if new_word.lower() not in st.session_state.stop_words:
            st.session_state.stop_words.append(new_word.lower())
            st.rerun()
    current_words = st.multiselect("Daftar Stop Words:", options=st.session_state.stop_words, default=st.session_state.stop_words)
    if len(current_words) != len(st.session_state.stop_words):
        st.session_state.stop_words = current_words
        st.rerun()
    if st.button("Simpan & Tutup", type="primary"):
        st.rerun()

# --- MAIN APP UI ---
with st.sidebar:
    st.title("⚙️ Pengaturan")
    theme_mode = st.segmented_control("Tema Tampilan", ["Light", "Dark"], default="Light")
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
    if st.button("Kelola Stop Words", use_container_width=True):
        manage_stopwords_dialog()

col_logo, col_title = st.columns([1, 10])
with col_logo: st.write("## 📝")
with col_title:
    st.title("AnaText 2.0")
    st.write("Analisis Teks, Sentimen, N-Gram & NER Berbasis AI")

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    api_key = ""
client = OpenAI(api_key=api_key) if api_key else None
MODEL_NAME = "gpt-4o"

container_input = st.container()
with container_input:
    tab_upload, tab_text = st.tabs(["📂 Unggah Dokumen", "✍️ Teks Langsung"])
    input_text_list = []
    
    with tab_upload:
        st.info("Mendukung format .txt, .csv, .xlsx")
        uploaded_file = st.file_uploader("Klik atau Seret File ke Sini", type=['csv', 'xlsx', 'txt'])
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                try: df_upload = pd.read_csv(uploaded_file, encoding='utf-8')
                except: df_upload = pd.read_csv(uploaded_file, encoding='latin-1')
            elif uploaded_file.name.endswith('.xlsx'): df_upload = pd.read_excel(uploaded_file)
            else:
                try: raw_text = uploaded_file.read().decode("utf-8")
                except: raw_text = uploaded_file.read().decode("latin-1")
                df_upload = pd.DataFrame(raw_text.splitlines(), columns=['Teks'])
            
            possible_cols = [c for c in df_upload.columns if df_upload[c].dtype == 'object']
            if possible_cols:
                text_col = st.selectbox("Konfirmasi Kolom Teks:", possible_cols)
                input_text_list = df_upload[text_col].dropna().astype(str).tolist()
            else: st.error("File tidak memiliki kolom teks yang valid.")

    with tab_text:
        direct_text = st.text_area("Tempelkan teks di sini...", height=200)
        if direct_text: input_text_list = [t for t in direct_text.split('\n') if t.strip()]

# --- PROSES ANALISIS ---
if st.button("🚀 Lakukan Analisis", type="primary"):
    if not input_text_list:
        st.warning("Mohon masukkan data teks terlebih dahulu.")
    elif not client:
        st.error("API Key belum dikonfigurasi di Secrets.")
    else:
        with st.spinner("Sedang memproses..."):
            df = pd.DataFrame(input_text_list, columns=['Teks_Asli'])
            
            # 1. Preprocessing
            factory = StemmerFactory()
            stemmer = factory.create_stemmer() if (language == "Indonesia" and check_lemma) else None
            
            bar = st.progress(0, text="Preprocessing...")
            clean_results = []
            for idx, text in enumerate(df['Teks_Asli']):
                cleaned = clean_text(text, check_sw, check_lemma, check_lower, st.session_state.stop_words, stemmer)
                clean_results.append(cleaned)
                bar.progress((idx+1)/len(df))
            df['Teks_Clean'] = clean_results
            bar.empty()

            # 2. Topic Modeling & TF-IDF
            df = df[df['Teks_Clean'].str.strip() != ""]
            actual_clusters = min(num_clusters_input, len(df))
            vectorizer = TfidfVectorizer(max_features=2000)
            tfidf_matrix = vectorizer.fit_transform(df['Teks_Clean'])
            feature_names = vectorizer.get_feature_names_out()
            kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10).fit(tfidf_matrix)
            df['Cluster_ID'] = kmeans.labels_
            
            cluster_names = {}
            for i in range(actual_clusters):
                centroid = kmeans.cluster_centers_[i]
                top_indices = centroid.argsort()[-5:][::-1]
                top_words = [feature_names[ind] for ind in top_indices]
                if top_words: cluster_names[i] = get_topic_name_ai(client, MODEL_NAME, top_words)
                else: cluster_names[i] = f"Topik {i+1}"
            df['Topik'] = df['Cluster_ID'].map(cluster_names)

            # 3. Sentimen AI
            df['Sentimen'] = get_sentiment_ai(client, MODEL_NAME, df['Teks_Asli'].tolist())

            # 4. NER Analysis (NEW FEATURE)
            # Kita gunakan Teks_Asli untuk NER karena case sensitive penting untuk nama
            ner_results = get_ner_ai(client, MODEL_NAME, df['Teks_Asli'].tolist())
            st.session_state.ner_data = ner_results

            st.session_state.data = df
            st.session_state.vectorizer = vectorizer
            st.session_state.tfidf_matrix = tfidf_matrix
            st.session_state.analysis_done = True
            st.rerun()

# --- DASHBOARD HASIL ---
if st.session_state.analysis_done and st.session_state.data is not None:
    df = st.session_state.data
    ner_data = st.session_state.ner_data
    
    st.write("---")
    st.subheader("📊 Insight Dashboard")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Dokumen", len(df))
    sentiment_counts = df['Sentimen'].value_counts()
    top_sent = sentiment_counts.idxmax() if not sentiment_counts.empty else "-"
    m2.metric("Dominasi Sentimen", top_sent)
    m3.metric("Jumlah Topik", df['Topik'].nunique())
    
    # Hitung total entitas
    total_entitas = len(ner_data["Person"]) + len(ner_data["Organization"]) + len(ner_data["Location"])
    m4.metric("Total Entitas (NER)", total_entitas)

    # Tabs Updated
    tab_sum, tab_sent, tab_topic, tab_ngram, tab_ner = st.tabs([
        "Ringkasan Eksekutif", "Analisis Sentimen", "Klaster Topik", "N-Gram & Frasa", "Entitas (NER)"
    ])

    with tab_sum:
        st.info("Insight dihasilkan oleh AI berdasarkan keseluruhan data.")
        if st.button("Generate AI Summary"):
            with st.spinner("Menulis laporan..."):
                summary_prompt = f"Data: {len(df)} teks. Sentimen: {sentiment_counts.to_dict()}. Topik: {df['Topik'].value_counts().head(3).index.tolist()}. Entitas Utama: {ner_data['Person'][:5]}. Buat ringkasan eksekutif."
                try:
                    res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role":"user", "content": summary_prompt}])
                    st.markdown(res.choices[0].message.content)
                except: st.error("Gagal generate summary.")

    with tab_sent:
        col_chart, col_table = st.columns([1, 2])
        with col_chart:
            fig_sent = px.pie(names=sentiment_counts.index, values=sentiment_counts.values, hole=0.5, color=sentiment_counts.index, color_discrete_map={'Positif':'#34d399', 'Negatif':'#f87171', 'Netral':'#9ca3af'})
            st.plotly_chart(fig_sent, use_container_width=True)
        with col_table:
            st.dataframe(df[['Teks_Asli', 'Sentimen']], use_container_width=True, height=400)

    with tab_topic:
        col_t1, col_t2 = st.columns([2, 1])
        with col_t1:
            topic_count = df['Topik'].value_counts().reset_index()
            topic_count.columns = ['Topik', 'Jumlah']
            st.plotly_chart(px.bar(topic_count, x='Jumlah', y='Topik', orientation='h', color='Jumlah'), use_container_width=True)
        with col_t2: st.table(topic_count)

    # --- TAB BARU: N-GRAM ---
    with tab_ngram:
        st.write("Analisis Frasa yang sering muncul bersamaan.")
        col_bi, col_tri = st.columns(2)
        
        with col_bi:
            st.subheader("Bigram (2 Kata)")
            bigrams = get_ngrams(df['Teks_Clean'], n=2, top_k=10)
            if bigrams:
                df_bi = pd.DataFrame(bigrams, columns=['Frasa', 'Frekuensi'])
                fig_bi = px.bar(df_bi, x='Frekuensi', y='Frasa', orientation='h', title="Top Bigram", color='Frekuensi')
                fig_bi.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_bi, use_container_width=True)
            else:
                st.info("Data tidak cukup untuk Bigram.")

        with col_tri:
            st.subheader("Trigram (3 Kata)")
            trigrams = get_ngrams(df['Teks_Clean'], n=3, top_k=10)
            if trigrams:
                df_tri = pd.DataFrame(trigrams, columns=['Frasa', 'Frekuensi'])
                fig_tri = px.bar(df_tri, x='Frekuensi', y='Frasa', orientation='h', title="Top Trigram", color='Frekuensi')
                fig_tri.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_tri, use_container_width=True)
            else:
                st.info("Data tidak cukup untuk Trigram.")
        
        st.write("---")
        st.write("**Word Cloud (Unigram)**")
        all_text = " ".join(df['Teks_Clean'])
        if all_text.strip():
            wc = WordCloud(width=800, height=300, background_color='white').generate(all_text)
            plt.figure(figsize=(10, 4))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)

    # --- TAB BARU: NER ---
    with tab_ner:
        st.write("Deteksi Entitas Penting (Orang, Organisasi, Lokasi) menggunakan AI.")
        
        # Helper function untuk bikin chart NER
        def plot_ner_category(data_list, title, color_seq):
            if not data_list:
                st.info(f"Tidak ditemukan entitas {title}.")
                return
            
            # Normalisasi nama (Title Case) agar 'jokowi' dan 'Jokowi' dianggap sama
            clean_list = [item.title() for item in data_list]
            df_ner = pd.Series(clean_list).value_counts().reset_index()
            df_ner.columns = ['Entitas', 'Jumlah']
            df_ner = df_ner.head(10) # Top 10 only
            
            fig = px.bar(df_ner, x='Jumlah', y='Entitas', orientation='h', title=f"Top {title}", color_discrete_sequence=[color_seq])
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

        col_n1, col_n2, col_n3 = st.columns(3)
        
        with col_n1:
            plot_ner_category(ner_data['Person'], "Person (Orang)", "#3b82f6") # Blue
        with col_n2:
            plot_ner_category(ner_data['Organization'], "Organization (Instansi)", "#ef4444") # Red
        with col_n3:
            plot_ner_category(ner_data['Location'], "Location (Lokasi)", "#10b981") # Green
            
        with st.expander("Lihat Data Mentah NER"):
            st.json(ner_data)
