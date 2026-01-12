import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from openai import OpenAI
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="AnaText - AI Text Analysis",
    page_icon="📊",
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

SENTIMENT_COLORS = {
    'Positif': '#28a745',
    'Negatif': '#dc3545',
    'Netral': '#ffc107',
    'Error': '#6c757d'
}

# --- STATE MANAGEMENT ---
if 'stop_words' not in st.session_state:
    st.session_state.stop_words = list(set(default_stopwords_id + default_stopwords_en))
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'topic_details' not in st.session_state:
    st.session_state.topic_details = []

# --- CSS CUSTOM (SMART CONTRAST) ---
def inject_custom_css(mode):
    # Definisi Warna Berbasis Mode
    if mode == 'Dark':
        bg_color = "#0e1117"
        sidebar_bg = "#262730"
        text_color = "#ffffff"
        text_secondary = "#e0e0e0"
        input_bg = "#41444C"
        border_col = "#555"
        btn_txt = "#ffffff"
    else:
        bg_color = "#ffffff"
        sidebar_bg = "#f0f2f6"
        text_color = "#000000"
        text_secondary = "#31333F"
        input_bg = "#ffffff"
        border_col = "#d1d5db"
        btn_txt = "#ffffff" # Tombol primary biasanya tetap teks putih

    st.markdown(f"""
    <style>
        /* Global Background & Text */
        .stApp {{ background-color: {bg_color}; color: {text_color}; }}
        
        /* Sidebar Specific */
        [data-testid="stSidebar"] {{
            background-color: {sidebar_bg};
        }}
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
            color: {text_color} !important;
        }}
        
        /* Text Elements Contrast */
        p, h1, h2, h3, h4, h5, h6, li, span, div, label {{
            color: {text_color};
        }}
        .stMarkdown {{ color: {text_color} !important; }}
        
        /* Inputs (Selectbox, Text Input) */
        .stTextInput > div > div > input {{
            color: {text_color};
            background-color: {input_bg};
        }}
        .stSelectbox > div > div {{
            background-color: {input_bg};
            color: {text_color};
        }}
        /* Fix dropdown text visibility */
        div[data-baseweb="select"] > div {{
            background-color: {input_bg};
            color: {text_color};
        }}
        
        /* Drag & Drop Area */
        [data-testid='stFileUploader'] {{
            background-color: {sidebar_bg};
            border: 2px dashed #4c7bf4;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        
        /* Buttons */
        .stButton button {{
            font-weight: bold;
            color: {btn_txt} !important;
        }}
        
        /* Footer Styling */
        .footer-text {{
            text-align: center;
            font-size: 12px;
            color: {text_secondary};
            margin-top: 50px;
            border-top: 1px solid {border_col};
            padding-top: 10px;
        }}

        /* Table Styling fixes */
        [data-testid="stDataFrame"] {{
             border: 1px solid {border_col};
        }}
    </style>
    """, unsafe_allow_html=True)
    
    return "plotly_dark" if mode == 'Dark' else "plotly_white"

# --- FUNGSI UTAMA ---

def clean_text(text, remove_sw, use_lemma, case_folding, stopwords_list, stemmer):
    if not isinstance(text, str): return ""
    if case_folding: text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    if remove_sw: tokens = [word for word in tokens if word not in stopwords_list]
    if use_lemma and stemmer: tokens = [stemmer.stem(word) for word in tokens]
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
            if "positif" in sentiment.lower(): sentiment = "Positif"
            elif "negatif" in sentiment.lower(): sentiment = "Negatif"
            else: sentiment = "Netral"
            results.append(sentiment)
        except: results.append("Error")
        if i % 5 == 0 or i == total - 1: progress_bar.progress((i + 1) / total)
    progress_bar.empty()
    return results

def get_topic_name_ai(client, model, keywords):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Berikan nama topik singkat (2-4 kata) yang merepresentasikan keywords ini."},
                {"role": "user", "content": f"Keywords: {', '.join(keywords)}"}
            ]
        )
        return response.choices[0].message.content.replace('"', '')
    except: return "Topik Umum"

def generate_text_network(topic_details, theme_mode):
    G = nx.Graph()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#F1948A', '#82E0AA', '#85C1E9']
    
    labels = {}
    
    for idx, detail in enumerate(topic_details):
        topic_name = detail['Topik']
        keywords = detail['Keywords'].split(', ')
        cluster_color = colors[idx % len(colors)]
        
        G.add_node(topic_name, size=2000, color=cluster_color, type='topic')
        labels[topic_name] = topic_name
        
        for kw in keywords:
            if not G.has_node(kw):
                G.add_node(kw, size=500, color=cluster_color, type='keyword')
                labels[kw] = kw
            G.add_edge(topic_name, kw)

    plt.figure(figsize=(12, 8), facecolor='#0e1117' if theme_mode=='Dark' else '#ffffff')
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    final_colors = [G.nodes[n]['color'] for n in G.nodes()]
    node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color=final_colors, alpha=0.9, node_size=node_sizes)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray')
    
    font_color = 'white' if theme_mode=='Dark' else 'black'
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color=font_color, font_weight='bold')
    
    plt.axis('off')
    return plt

# --- MODAL STOPWORDS ---
def show_stopwords_manager():
    col1, col2 = st.columns([3, 1])
    with col1: new_word = st.text_input("Tambah kata:", label_visibility="collapsed", placeholder="Ketik kata...")
    with col2: 
        if st.button("Tambah"):
            if new_word and new_word.lower() not in st.session_state.stop_words:
                st.session_state.stop_words.append(new_word.lower()); st.rerun()
    current = st.multiselect("Daftar Stop Words:", st.session_state.stop_words, default=st.session_state.stop_words)
    if len(current) != len(st.session_state.stop_words): st.session_state.stop_words = current; st.rerun()

if hasattr(st, "dialog"):
    @st.dialog("Kelola Stop Words")
    def open_stopwords_modal(): show_stopwords_manager(); st.button("Tutup", on_click=st.rerun)
else:
    def open_stopwords_modal(): pass

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Pengaturan")
    theme_mode = st.radio("Tema Tampilan", ["Light", "Dark"], horizontal=True)
    plotly_template = inject_custom_css(theme_mode)
    
    st.divider()
    st.subheader("Bahasa & Teks")
    language = st.selectbox("Bahasa Teks", ["Indonesia", "Inggris"])
    text_type = st.selectbox("Tipe Teks", ["Umum", "Ulasan Produk", "Berita/Artikel", "Media Sosial", "Akademik"])
    
    st.divider()
    st.subheader("🤖 Model AI")
    num_clusters_input = st.slider("Jumlah Topik (Klaster)", 2, 10, 5)
    
    st.divider()
    st.subheader("🔧 Preprocessing")
    check_sw = st.checkbox("Hapus Stop Words", value=True, help="Menghapus kata-kata umum yang sering muncul tapi minim makna (contoh: yang, di, ke, dari).")
    check_lemma = st.checkbox("Aktifkan Lemmatization", value=True, help="Mengubah kata berimbuhan menjadi kata dasar (contoh: 'memakan' menjadi 'makan') menggunakan algoritma Sastrawi.")
    check_lower = st.checkbox("Case Folding (lowercase)", value=True, help="Mengubah semua huruf dalam teks menjadi huruf kecil agar seragam.")
    
    if hasattr(st, "dialog"):
        if st.button("Kelola Stop Words", use_container_width=True): open_stopwords_modal()
    else:
        with st.expander("Kelola Stop Words"): show_stopwords_manager()
    
    # REVISI 4: FOOTER DI SIDEBAR
    st.markdown("---")
    st.markdown('<div class="footer-text">Developed by <b>Suwarno</b><br>Powered by <b>Open AI</b></div>', unsafe_allow_html=True)

# --- MAIN UI ---
col_logo, col_title = st.columns([1, 12])
with col_title: st.title("📊 AnaText") 
st.write("Platform Analisis Teks Berbasis AI")

try: api_key = st.secrets["OPENAI_API_KEY"]
except: api_key = "" 
client = OpenAI(api_key=api_key) if api_key else None
MODEL_NAME = "gpt-4o"

# --- INPUT ---
container_input = st.container()
with container_input:
    tab_upload, tab_text = st.tabs(["📂 Unggah Dokumen", "✍️ Teks Langsung"])
    input_text_list = []
    
    with tab_upload:
        st.info("Limit file: 10 MB. Format: .csv, .xlsx, .txt")
        uploaded_file = st.file_uploader("Upload File", type=['csv', 'xlsx', 'txt'])
        if uploaded_file:
            if uploaded_file.size > 10 * 1024 * 1024: st.error("File terlalu besar (>10MB).")
            else:
                if uploaded_file.name.endswith('.csv'):
                    try: df_u = pd.read_csv(uploaded_file, encoding='utf-8')
                    except: df_u = pd.read_csv(uploaded_file, encoding='latin-1')
                elif uploaded_file.name.endswith('.xlsx'): df_u = pd.read_excel(uploaded_file)
                else:
                    try: txt = uploaded_file.read().decode('utf-8')
                    except: txt = uploaded_file.read().decode('latin-1')
                    df_u = pd.DataFrame(txt.splitlines(), columns=['Teks'])
                
                cols = [c for c in df_u.columns if df_u[c].dtype == 'object']
                if cols:
                    t_col = st.selectbox("Pilih Kolom Teks:", cols)
                    input_text_list = df_u[t_col].dropna().astype(str).tolist()
                else: st.error("Tidak ada kolom teks.")

    with tab_text:
        dt = st.text_area("Tempel teks...", height=150)
        if dt: input_text_list = [t for t in dt.split('\n') if t.strip()]

# --- PROCESSING ---
if st.button("🚀 Lakukan Analisis", type="primary"):
    if not input_text_list: st.warning("Data kosong.")
    elif not client: st.error("API Key missing.")
    else:
        with st.spinner("Sedang memproses..."):
            df = pd.DataFrame(input_text_list, columns=['Teks_Asli'])
            factory = StemmerFactory()
            stemmer = factory.create_stemmer() if (language == "Indonesia" and check_lemma) else None
            
            clean_res = []
            pb = st.progress(0)
            for i, t in enumerate(df['Teks_Asli']):
                clean_res.append(clean_text(t, check_sw, check_lemma, check_lower, st.session_state.stop_words, stemmer))
                if i % 10 == 0: pb.progress((i+1)/len(df))
            pb.empty()
            
            df['Teks_Clean'] = clean_res
            df = df[df['Teks_Clean'].str.strip() != ""]
            
            # Clustering
            k = min(num_clusters_input, len(df))
            if k < 2: k = 1
            
            vec = TfidfVectorizer(max_features=2000)
            tfidf = vec.fit_transform(df['Teks_Clean'])
            feats = vec.get_feature_names_out()
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(tfidf)
            df['Cluster_ID'] = kmeans.labels_
            
            topic_map = {}
            topic_data_list = []
            
            for i in range(k):
                center = kmeans.cluster_centers_[i]
                top_idx = center.argsort()[-10:][::-1] 
                top_w = [feats[x] for x in top_idx]
                
                label = get_topic_name_ai(client, MODEL_NAME, top_w[:5]) if top_w else f"Topik {i+1}"
                topic_map[i] = label
                
                topic_data_list.append({
                    'Nomor': i+1,
                    'Topik': label,
                    'Keywords': ", ".join(top_w)
                })
            
            df['Topik'] = df['Cluster_ID'].map(topic_map)
            df['Sentimen'] = get_sentiment_ai(client, MODEL_NAME, df['Teks_Asli'].tolist())
            
            st.session_state.data = df
            st.session_state.topic_details = topic_data_list
            st.session_state.vectorizer = vec
            st.session_state.tfidf_matrix = tfidf
            st.session_state.analysis_done = True
            st.rerun()

# --- DASHBOARD ---
if st.session_state.analysis_done and st.session_state.data is not None:
    df = st.session_state.data
    st.write("---")
    st.subheader("📊 Insight Dashboard")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Dokumen", len(df))
    sc = df['Sentimen'].value_counts()
    m2.metric("Dominasi Sentimen", sc.idxmax() if not sc.empty else "-")
    m3.metric("Jumlah Topik", df['Topik'].nunique())
    
    # REVISI 2: TAB ICONS
    t1, t2, t3, t4, t5 = st.tabs(["📝 Ringkasan", "🎭 Sentimen", "📂 Topik", "🔠 Kata Kunci", "🌐 Network Analysis"])
    
    with t1:
        st.info("Tekan tombol di bawah untuk mendapatkan analisis mendalam.")
        if st.button("Generate Comprehensive Summary"):
            with st.spinner("AI sedang menyusun laporan lengkap..."):
                try:
                    topics_str = "\n".join([f"- {t['Topik']}: {t['Keywords']}" for t in st.session_state.topic_details])
                    # REVISI 5: Prompt Analisis Network
                    prompt = f"""
                    Bertindaklah sebagai Data Analyst Senior. Analisis data berikut:
                    KONTEKS: Tipe Teks: {text_type}. Total Data: {len(df)}. Sentimen: {sc.to_dict()}.
                    
                    JARINGAN TOPIK & KEYWORD (NETWORK ANALYSIS):
                    Sistem telah mengelompokkan teks ke dalam topik-topik berikut beserta kata kuncinya (Keywords saling terhubung membentuk klaster):
                    {topics_str}
                    
                    TUGAS: Buat Executive Summary (Bahasa Indonesia).
                    1. **Gambaran Umum**: Performa sentimen.
                    2. **Analisis Sentimen**: Mengapa sentimen dominan terjadi?
                    3. **Interpretasi Network Analysis**: Jelaskan pola hubungan antar kata dan topik yang terbentuk dari data di atas. Bagaimana kata-kata kunci dalam satu topik saling berkaitan membentuk makna?
                    4. **Kesimpulan & Rekomendasi**.
                    """
                    res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role":"user", "content": prompt}])
                    st.markdown(res.choices[0].message.content)
                except Exception as e: st.error(f"Gagal generate summary: {str(e)}")

    with t2:
        c1, c2 = st.columns([1, 2])
        with c1:
            fig = px.pie(values=sc.values, names=sc.index, hole=0.5, color=sc.index, 
                         color_discrete_map=SENTIMENT_COLORS, template=plotly_template)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            f_s = st.multiselect("Filter:", df['Sentimen'].unique(), default=df['Sentimen'].unique())
            def color_row(v):
                if v=='Positif': return 'background-color: #28a745; color: white'
                if v=='Negatif': return 'background-color: #dc3545; color: white'
                if v=='Netral': return 'background-color: #ffc107; color: black'
                return ''
            st.dataframe(df[df['Sentimen'].isin(f_s)][['Teks_Asli','Sentimen']].style.map(color_row, subset=['Sentimen']), use_container_width=True)

    with t3:
        tc = df['Topik'].value_counts().reset_index()
        tc.columns = ['Topik', 'Jumlah']
        fig_bar = px.bar(tc, x='Jumlah', y='Topik', orientation='h', color='Jumlah', template=plotly_template)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.write("#### 📋 Detail Kata Kunci per Topik")
        # REVISI 3: Desain Tabel Lebih Menarik (Interactive, No Index)
        df_topics = pd.DataFrame(st.session_state.topic_details)
        st.dataframe(
            df_topics, 
            hide_index=True, 
            use_container_width=True,
            column_config={
                "Nomor": st.column_config.NumberColumn("No.", width="small"),
                "Topik": st.column_config.TextColumn("Nama Topik", width="medium"),
                "Keywords": st.column_config.TextColumn("Kata Kunci (Keywords)", width="large")
            }
        )

    with t4:
        txt_all = " ".join(df['Teks_Clean'])
        if txt_all.strip():
            wc = WordCloud(width=800, height=400, background_color='black' if theme_mode=='Dark' else 'white').generate(txt_all)
            plt.figure(figsize=(10, 5), facecolor='k' if theme_mode=='Dark' else 'w')
            plt.imshow(wc, interpolation='bilinear'); plt.axis("off")
            st.pyplot(plt)
        
        sum_tfidf = st.session_state.tfidf_matrix.sum(axis=0)
        words = [(word, sum_tfidf[0, idx]) for word, idx in st.session_state.vectorizer.vocabulary_.items()]
        words = sorted(words, key=lambda x: x[1], reverse=True)[:10]
        df_k = pd.DataFrame(words, columns=["Kata", "Skor"])
        fig_k = px.bar(df_k, x="Skor", y="Kata", orientation='h', template=plotly_template)
        fig_k.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_k, use_container_width=True)

    with t5:
        st.subheader("🕸️ Text Network Analysis")
        st.write("Visualisasi hubungan antara topik (node besar) dan kata kunci dominan (node kecil).")
        if st.session_state.topic_details:
            fig_net = generate_text_network(st.session_state.topic_details, theme_mode)
            st.pyplot(fig_net)
        else:
            st.warning("Data topik belum tersedia.")

    st.divider()
    st.download_button("📥 Unduh CSV", df.to_csv(index=False).encode('utf-8'), "analisis_anatext.csv", "text/csv")
