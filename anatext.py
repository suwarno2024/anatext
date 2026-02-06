import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from openai import OpenAI
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import json
from collections import Counter
from datetime import datetime

APP_VERSION = "2.0.0-enhanced"

# ============================================================
#                  KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="AnaText - AI Text Analysis",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
#              DEFINISI STOPWORDS & CONSTANTS
# ============================================================
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

TOPIC_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
    '#F7DC6F', '#BB8FCE', '#F1948A', '#82E0AA', '#85C1E9'
]

# ============================================================
#                   STATE MANAGEMENT
# ============================================================
if 'stop_words' not in st.session_state:
    st.session_state.stop_words = list(set(default_stopwords_id + default_stopwords_en))
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'topic_details' not in st.session_state:
    st.session_state.topic_details = []
if 'ner_results' not in st.session_state:
    st.session_state.ner_results = None
if 'summary_cache' not in st.session_state:
    st.session_state.summary_cache = None

# ============================================================
#                  CSS & THEME ENGINE
# ============================================================
def inject_custom_css(mode):
    if mode == 'Dark':
        bg_color = "#0e1117"
        sidebar_bg = "#1a1d24"
        text_color = "#e8e8e8"
        text_secondary = "#9ca3af"
        input_bg = "#2d3139"
        border_col = "#3d4150"
        btn_txt = "#ffffff"
        card_bg = "#1a1d24"
        card_border = "#2d3139"
        accent = "#4facfe"
        metric_bg = "#1e2128"
        hover_bg = "#262b36"
        tab_bg = "#1a1d24"
        success_bg = "#0d3320"
        info_bg = "#0d2847"
        warning_bg = "#3d2e00"
    else:
        bg_color = "#f8f9fb"
        sidebar_bg = "#ffffff"
        text_color = "#1a1a2e"
        text_secondary = "#64748b"
        input_bg = "#ffffff"
        border_col = "#e2e8f0"
        btn_txt = "#ffffff"
        card_bg = "#ffffff"
        card_border = "#e2e8f0"
        accent = "#3b82f6"
        metric_bg = "#f1f5f9"
        hover_bg = "#f8fafc"
        tab_bg = "#ffffff"
        success_bg = "#ecfdf5"
        info_bg = "#eff6ff"
        warning_bg = "#fffbeb"

    st.markdown(f"""
    <style>
        /* ---- BASE ---- */
        .stApp {{ background-color: {bg_color}; color: {text_color}; }}
        [data-testid="stSidebar"] {{ background-color: {sidebar_bg}; border-right: 1px solid {border_col}; }}
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {{ color: {text_color} !important; }}
        p, h1, h2, h3, h4, h5, h6, li, span, div, label {{ color: {text_color}; }}
        .stMarkdown {{ color: {text_color} !important; }}

        /* ---- INPUTS ---- */
        .stTextInput > div > div > input {{ color: {text_color}; background-color: {input_bg}; border: 1px solid {border_col}; border-radius: 8px; }}
        .stTextArea textarea {{ color: {text_color}; background-color: {input_bg}; border: 1px solid {border_col}; border-radius: 8px; }}
        div[data-baseweb="select"] > div {{ background-color: {input_bg}; color: {text_color}; border-radius: 8px; }}

        /* ---- UPLOAD AREA ---- */
        [data-testid='stFileUploader'] {{
            background-color: {card_bg};
            border: 2px dashed {accent};
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
        }}
        [data-testid='stFileUploader']:hover {{ border-color: #00f2fe; }}

        /* ---- BUTTONS ---- */
        .stButton button {{
            font-weight: 600;
            color: {btn_txt} !important;
            border-radius: 8px;
            transition: all 0.2s ease;
            border: none;
        }}
        .stButton button:hover {{ transform: translateY(-1px); box-shadow: 0 4px 12px rgba(79,172,254,0.3); }}

        /* ---- METRICS CARDS ---- */
        [data-testid="stMetric"] {{
            background: {metric_bg};
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        [data-testid="stMetric"] label {{ color: {text_secondary} !important; font-size: 0.85rem; font-weight: 500; }}
        [data-testid="stMetric"] [data-testid="stMetricValue"] {{ color: {text_color} !important; font-weight: 700; }}

        /* ---- TABS ---- */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background: {metric_bg};
            border-radius: 10px;
            padding: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 500;
            color: {text_secondary};
        }}
        .stTabs [aria-selected="true"] {{
            background: {accent} !important;
            color: white !important;
            font-weight: 600;
        }}

        /* ---- TABLES ---- */
        [data-testid="stDataFrame"] {{ border: 1px solid {border_col}; border-radius: 8px; overflow: hidden; }}

        /* ---- DIVIDER ---- */
        hr {{ border-color: {border_col}; opacity: 0.5; }}

        /* ---- CUSTOM CARDS ---- */
        .custom-card {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }}

        .stat-card {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            transition: all 0.2s ease;
        }}
        .stat-card:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,0.08); transform: translateY(-2px); }}
        .stat-card .stat-value {{ font-size: 2rem; font-weight: 800; line-height: 1.2; }}
        .stat-card .stat-label {{ font-size: 0.85rem; color: {text_secondary}; margin-top: 4px; font-weight: 500; }}

        .ner-card {{
            background: {card_bg};
            border-left: 4px solid;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 8px;
        }}

        /* ---- FOOTER ---- */
        .footer-text {{
            text-align: center; font-size: 12px; color: {text_secondary};
            margin-top: 50px; border-top: 1px solid {border_col}; padding-top: 10px;
        }}

        /* ---- EXPANDER ---- */
        .streamlit-expanderHeader {{ background-color: {card_bg}; border-radius: 8px; }}

        /* ---- SCROLLBAR ---- */
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-track {{ background: {bg_color}; }}
        ::-webkit-scrollbar-thumb {{ background: {border_col}; border-radius: 3px; }}
    </style>
    """, unsafe_allow_html=True)

    return "plotly_dark" if mode == 'Dark' else "plotly_white"


# ============================================================
#                   HEADER COMPONENT
# ============================================================
def render_elegant_header(mode):
    title_color = "#ffffff" if mode == 'Dark' else "#1a1a2e"
    subtitle_color = "#9ca3af" if mode == 'Dark' else "#64748b"
    badge_bg = "rgba(79,172,254,0.15)" if mode == 'Dark' else "rgba(59,130,246,0.1)"
    badge_text = "#4facfe" if mode == 'Dark' else "#3b82f6"

    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 36px; padding-top: 16px;">
        <div style="
            display: inline-flex; align-items: center; justify-content: center;
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            width: 72px; height: 72px; border-radius: 18px;
            box-shadow: 0 8px 24px rgba(79,172,254,0.35);
            margin-bottom: 12px;
        "><span style="font-size: 36px;">💡</span></div>
        <h1 style="
            color: {title_color}; font-family: 'Inter','Helvetica Neue',sans-serif;
            font-weight: 800; font-size: 2.6rem; margin: 0; letter-spacing: -1px;
        ">AnaText AI</h1>
        <p style="color: {subtitle_color}; font-size: 1.05rem; margin-top: 6px; font-weight: 400; letter-spacing: 0.3px;">
            Platform Analisis Teks Berbasis AI
        </p>
        <div style="margin-top: 10px;">
            <span style="background: {badge_bg}; color: {badge_text}; padding: 4px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 600;">
                ✨ Sentiment · Clustering · NER · N-Gram · Network
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
#              CORE ANALYSIS FUNCTIONS
# ============================================================

def clean_text(text, remove_sw, use_lemma, case_folding, stopwords_list, stemmer):
    """Membersihkan teks: lowering, punctuation removal, stopwords, stemming."""
    if not isinstance(text, str):
        return ""
    if case_folding:
        text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)          # hapus URL
    text = re.sub(r'@\w+', '', text)                       # hapus mention
    text = re.sub(r'#\w+', '', text)                       # hapus hashtag (simbol)
    text = re.sub(r'\d+', '', text)                         # hapus angka
    text = re.sub(r'[^\w\s]', '', text)                     # hapus punctuation
    text = re.sub(r'\s+', ' ', text).strip()                # hapus spasi berlebih
    tokens = text.split()
    if remove_sw:
        tokens = [w for w in tokens if w not in stopwords_list]
    if use_lemma and stemmer:
        tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)


# 1. Analisis Sentimen (AI) — dengan batching untuk efisiensi
def get_sentiment_ai(client, model, text_list, batch_size=5):
    """Menganalisis sentimen secara batch untuk mengurangi jumlah API call."""
    results = []
    progress_bar = st.progress(0)
    total = len(text_list)

    for i in range(0, total, batch_size):
        batch = text_list[i:i + batch_size]
        batch_texts = []
        for idx, text in enumerate(batch):
            if not text.strip():
                batch_texts.append(f"{idx + 1}. [kosong]")
            else:
                truncated = text[:500]
                batch_texts.append(f"{idx + 1}. {truncated}")

        numbered_texts = "\n".join(batch_texts)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": (
                        "Anda adalah analis sentimen. Klasifikasikan sentimen setiap teks: "
                        "Positif, Negatif, atau Netral. "
                        "Jawab HANYA dengan format JSON array, contoh: [\"Positif\",\"Negatif\",\"Netral\"]. "
                        "Tanpa penjelasan apapun."
                    )},
                    {"role": "user", "content": f"Klasifikasikan sentimen teks berikut:\n{numbered_texts}"}
                ],
                temperature=0,
                max_tokens=200
            )
            raw = response.choices[0].message.content.strip()
            # Coba parse JSON
            try:
                batch_results = json.loads(raw)
                # Normalisasi
                normalized = []
                for s in batch_results:
                    s_lower = s.lower().strip()
                    if "positif" in s_lower or "positive" in s_lower:
                        normalized.append("Positif")
                    elif "negatif" in s_lower or "negative" in s_lower:
                        normalized.append("Negatif")
                    else:
                        normalized.append("Netral")
                # Pastikan jumlah sesuai batch
                while len(normalized) < len(batch):
                    normalized.append("Netral")
                results.extend(normalized[:len(batch)])
            except json.JSONDecodeError:
                # Fallback: parse per baris
                for text in batch:
                    if not text.strip():
                        results.append("Netral")
                    else:
                        try:
                            resp = client.chat.completions.create(
                                model=model,
                                messages=[
                                    {"role": "system", "content": "Klasifikasikan sentimen: Positif, Negatif, atau Netral. Jawab 1 kata saja."},
                                    {"role": "user", "content": text}
                                ],
                                temperature=0, max_tokens=10
                            )
                            s = resp.choices[0].message.content.strip().replace(".", "")
                            if "positif" in s.lower():
                                results.append("Positif")
                            elif "negatif" in s.lower():
                                results.append("Negatif")
                            else:
                                results.append("Netral")
                        except Exception:
                            results.append("Error")
        except Exception:
            results.extend(["Error"] * len(batch))

        progress_bar.progress(min((i + len(batch)) / total, 1.0))

    progress_bar.empty()
    return results[:total]


# 2. Penamaan Topik (AI)
def get_topic_name_ai(client, model, keywords):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Berikan nama topik singkat (2-4 kata) dari keyword ini. Jawab hanya nama topik, tanpa tanda kutip."},
                {"role": "user", "content": f"Keywords: {', '.join(keywords)}"}
            ],
            temperature=0.3, max_tokens=30
        )
        return response.choices[0].message.content.strip().replace('"', '').replace("'", "")
    except Exception:
        return "Topik Umum"


# 3. N-Gram Analysis
def get_ngrams(text_series, n=2, top_k=10):
    """Menghasilkan frekuensi N-Gram tertinggi."""
    try:
        vec = CountVectorizer(ngram_range=(n, n)).fit(text_series)
        bag_of_words = vec.transform(text_series)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:top_k]
    except ValueError:
        return []


# 4. NER Analysis (AI)
def get_ner_ai(client, model, text_full):
    """Mengekstrak Named Entities menggunakan GPT-4o."""
    try:
        text_sample = text_full[:15000]
        prompt = f"""
        Anda adalah Ahli Bahasa Indonesia dan Inggris. Tugas Anda adalah mengekstrak Entitas Penting (Named Entity Recognition) dari teks berikut.

        Kategorikan ke dalam 3 jenis:
        1. Person (Nama Tokoh/Orang)
        2. Organization (Nama Instansi/Lembaga/Perusahaan)
        3. Location (Nama Tempat/Kota/Negara/Lokasi)

        Output HARUS dalam format JSON murni tanpa markdown, dengan struktur:
        {{
            "Person": ["nama1", "nama2"],
            "Organization": ["org1", "org2"],
            "Location": ["loc1", "loc2"]
        }}

        Pastikan entitas unik (tidak duplikat) dan relevan.

        TEKS:
        {text_sample}
        """
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful NLP assistant that outputs only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"Person": [], "Organization": [], "Location": [], "Error": str(e)}


# 5. Comprehensive AI Summary (ENHANCED)
def generate_ai_summary(client, model, df, sc, topic_details, ner_results, text_type, language):
    """Menghasilkan Executive Summary yang komprehensif dan terstruktur."""
    topics_str = "\n".join([f"  - Topik '{t['Topik']}': kata kunci [{t['Keywords']}]" for t in topic_details])

    ner = ner_results or {}
    persons = ner.get('Person', [])[:8]
    orgs = ner.get('Organization', [])[:8]
    locs = ner.get('Location', [])[:8]

    # Hitung cross-tab sentimen per topik
    cross_data = []
    for topic in df['Topik'].unique():
        subset = df[df['Topik'] == topic]
        svc = subset['Sentimen'].value_counts().to_dict()
        cross_data.append(f"  - {topic}: Positif={svc.get('Positif', 0)}, Negatif={svc.get('Negatif', 0)}, Netral={svc.get('Netral', 0)}")
    cross_str = "\n".join(cross_data)

    # Hitung kata dominan
    all_words = " ".join(df['Teks_Clean']).split()
    top_words = Counter(all_words).most_common(15)
    top_words_str = ", ".join([f"{w} ({c}x)" for w, c in top_words])

    # Rasio sentimen
    total = len(df)
    pos_pct = round(sc.get('Positif', 0) / total * 100, 1) if total else 0
    neg_pct = round(sc.get('Negatif', 0) / total * 100, 1) if total else 0
    net_pct = round(sc.get('Netral', 0) / total * 100, 1) if total else 0

    # Bigram teratas
    bigrams = get_ngrams(df['Teks_Clean'], n=2, top_k=5)
    bigram_str = ", ".join([f"'{b[0]}' ({b[1]}x)" for b in bigrams]) if bigrams else "tidak cukup data"

    prompt = f"""Anda adalah **Senior Data Analyst** berpengalaman di bidang Text Mining & NLP.
Tulis sebuah **Laporan Analisis Eksekutif** yang komprehensif dan profesional berdasarkan data di bawah ini.

═══════════════ DATA INPUT ═══════════════

📊 METADATA
- Tipe Teks: {text_type}
- Bahasa: {language}
- Total Dokumen Dianalisis: {total}
- Tanggal Analisis: {datetime.now().strftime('%d %B %Y')}

🎭 DISTRIBUSI SENTIMEN
- Positif: {sc.get('Positif', 0)} dokumen ({pos_pct}%)
- Negatif: {sc.get('Negatif', 0)} dokumen ({neg_pct}%)
- Netral: {sc.get('Netral', 0)} dokumen ({net_pct}%)

📂 TOPIK & KATA KUNCI (dari K-Means Clustering + TF-IDF)
{topics_str}

📊 SENTIMEN PER TOPIK (Cross-Tabulation)
{cross_str}

🔠 KATA DOMINAN (Frekuensi Tertinggi)
{top_words_str}

🔗 FRASA DOMINAN (Bigram)
{bigram_str}

👤 ENTITAS TERDETEKSI (NER)
- Person: {', '.join(persons) if persons else 'tidak terdeteksi'}
- Organization: {', '.join(orgs) if orgs else 'tidak terdeteksi'}
- Location: {', '.join(locs) if locs else 'tidak terdeteksi'}

═══════════════ INSTRUKSI OUTPUT ═══════════════

Tulis laporan analisis dengan struktur berikut. Gunakan bahasa Indonesia yang profesional, data-driven, dan insightful.

ATURAN PENTING: Setiap bagian (section) di bawah ini WAJIB ditulis sepanjang **200 hingga 250 kata**. Jangan kurang dari 200 kata dan jangan lebih dari 250 kata per bagian. Tulis secara mendalam, elaboratif, dan kaya analisis untuk memenuhi target jumlah kata tersebut.

## 📋 Ringkasan Eksekutif
(200-250 kata) Tulis paragraf pembuka komprehensif yang merangkum seluruh temuan analisis. Sebutkan jumlah dokumen, distribusi sentimen dominan beserta persentasenya, topik-topik utama yang teridentifikasi, entitas kunci, dan pola linguistik yang menonjol. Berikan gambaran menyeluruh sehingga pembaca langsung memahami lanskap data tanpa harus membaca seluruh laporan.

## 🎭 Analisis Sentimen
(200-250 kata) Jelaskan distribusi sentimen secara detail dan dominansinya. Analisis MENGAPA sentimen tertentu dominan berdasarkan konteks topik dan kata kunci. Hubungkan sentimen dengan tipe teks ({text_type}). Berikan interpretasi proporsi sentimen (apakah sehat/mengkhawatirkan?). Bandingkan rasio positif-negatif dan diskusikan implikasinya. Jelaskan potensi faktor penyebab sentimen netral jika signifikan.

## 📂 Analisis Tematik (Topik)
(200-250 kata) Jelaskan setiap topik yang teridentifikasi beserta makna dan konteksnya berdasarkan kata kunci yang membentuknya. Identifikasi topik paling krusial/dominan dari segi jumlah dokumen. Jelaskan hubungan dan keterkaitan antar topik jika ada. Diskusikan apakah topik-topik tersebut saling mendukung atau bertentangan, dan apa implikasinya terhadap narasi keseluruhan data.

## 🔀 Sentimen × Topik (Cross-Analysis)
(200-250 kata) Analisis topik mana yang paling positif vs paling negatif berdasarkan data cross-tabulation. Identifikasi adakah topik kontroversial yang sentimennya terpolarisasi (campuran kuat positif dan negatif). Tentukan topik mana yang perlu perhatian dan prioritas penanganan. Berikan penjelasan mendalam tentang mengapa topik tertentu mendapat respons sentimen yang berbeda.

## 👤 Analisis Entitas (NER)
(200-250 kata) Jelaskan siapa saja tokoh/orang kunci yang terdeteksi dan apa relevansinya dalam konteks data. Analisis organisasi/lembaga yang muncul dan perannya. Diskusikan lokasi yang terdeteksi dan hubungannya dengan narasi data. Kaitkan entitas-entitas ini dengan topik dan sentimen yang ditemukan. Jelaskan implikasi kehadiran entitas-entitas tersebut.

## 🔗 Pola Linguistik
(200-250 kata) Interpretasikan kata-kata dominan dan frasa (bigram) yang paling sering muncul. Jelaskan apa yang bisa disimpulkan dari pola penggunaan kata dan frasa tersebut. Identifikasi pola bahasa apa yang muncul secara konsisten. Diskusikan apakah ada kosakata spesifik yang menunjukkan kecenderungan tertentu dalam data. Hubungkan temuan linguistik dengan topik dan sentimen.

## 📌 Kesimpulan & Rekomendasi Strategis
(200-250 kata) Tuliskan 3-5 poin kesimpulan kunci yang merangkum seluruh temuan. Berikan 3-5 rekomendasi aksi konkret dan spesifik berdasarkan temuan data. Tentukan prioritas tindakan dari yang paling mendesak. Sertakan justifikasi berbasis data untuk setiap rekomendasi yang diberikan.

Gunakan **bold** untuk poin penting. Pastikan setiap klaim didukung oleh data kuantitatif yang tersedia.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "Anda adalah konsultan data analytics senior. "
                    "Tulis laporan yang mendalam, terstruktur, dan actionable. "
                    "Selalu kaitkan temuan dengan data kuantitatif yang diberikan. "
                    "WAJIB: Setiap bagian/section harus ditulis sepanjang 200-250 kata. "
                    "Tidak boleh kurang dari 200 kata per bagian. Elaborasi secara mendalam."
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=6000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Gagal menghasilkan ringkasan: {str(e)}"


# 6. Network Graph (Enhanced with Plotly for interactivity)
def generate_text_network_plotly(topic_details, theme_mode):
    """Membuat network graph interaktif menggunakan Plotly."""
    G = nx.Graph()

    for idx, detail in enumerate(topic_details):
        topic_name = detail['Topik']
        keywords = detail['Keywords'].split(', ')
        color = TOPIC_COLORS[idx % len(TOPIC_COLORS)]
        G.add_node(topic_name, size=30, color=color, ntype='topic')
        for kw in keywords[:8]:
            if not G.has_node(kw):
                G.add_node(kw, size=12, color=color, ntype='keyword')
            G.add_edge(topic_name, kw)

    pos = nx.spring_layout(G, k=0.6, seed=42, iterations=50)

    # Edges
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(150,150,150,0.4)'),
        hoverinfo='none', mode='lines'
    )

    # Nodes
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_color.append(G.nodes[node]['color'])
        node_size.append(G.nodes[node]['size'])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        textfont=dict(size=9, color='white' if theme_mode == 'Dark' else 'black'),
        marker=dict(size=node_size, color=node_color, line=dict(width=1, color='rgba(255,255,255,0.3)'))
    )

    bg_color = '#0e1117' if theme_mode == 'Dark' else '#f8f9fb'
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        paper_bgcolor=bg_color,
                        plot_bgcolor=bg_color,
                        margin=dict(b=20, l=20, r=20, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    ))
    return fig


# 7. Matplotlib Network (fallback, preserved from original)
def generate_text_network(topic_details, theme_mode):
    G = nx.Graph()
    labels = {}
    for idx, detail in enumerate(topic_details):
        topic_name = detail['Topik']
        keywords = detail['Keywords'].split(', ')
        cluster_color = TOPIC_COLORS[idx % len(TOPIC_COLORS)]
        G.add_node(topic_name, size=2000, color=cluster_color, type='topic')
        labels[topic_name] = topic_name
        for kw in keywords:
            if not G.has_node(kw):
                G.add_node(kw, size=500, color=cluster_color, type='keyword')
                labels[kw] = kw
            G.add_edge(topic_name, kw)

    fig_bg = '#0e1117' if theme_mode == 'Dark' else '#ffffff'
    plt.figure(figsize=(12, 8), facecolor=fig_bg)
    pos = nx.spring_layout(G, k=0.5, seed=42)
    final_colors = [G.nodes[n]['color'] for n in G.nodes()]
    node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=final_colors, alpha=0.9, node_size=node_sizes)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray')
    font_color = 'white' if theme_mode == 'Dark' else 'black'
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color=font_color, font_weight='bold')
    plt.axis('off')
    return plt


# 8. Topic Cluster Scatter (PCA)
def generate_cluster_scatter(tfidf_matrix, labels, topic_names, template):
    """Visualisasi 2D cluster menggunakan PCA."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(tfidf_matrix.toarray())
    df_scatter = pd.DataFrame({
        'PC1': coords[:, 0],
        'PC2': coords[:, 1],
        'Topik': [topic_names.get(l, f"Topik {l}") for l in labels]
    })
    fig = px.scatter(
        df_scatter, x='PC1', y='PC2', color='Topik',
        color_discrete_sequence=TOPIC_COLORS,
        template=template,
        title="Distribusi Dokumen per Cluster (PCA 2D)"
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(height=450)
    return fig


# ============================================================
#              STOPWORDS MANAGER (MODAL)
# ============================================================
def show_stopwords_manager():
    col1, col2 = st.columns([3, 1])
    with col1:
        new_word = st.text_input("Tambah kata:", label_visibility="collapsed", placeholder="Ketik kata...")
    with col2:
        if st.button("Tambah"):
            if new_word and new_word.lower() not in st.session_state.stop_words:
                st.session_state.stop_words.append(new_word.lower())
                st.rerun()
    current = st.multiselect("Daftar Stop Words:", st.session_state.stop_words, default=st.session_state.stop_words)
    if len(current) != len(st.session_state.stop_words):
        st.session_state.stop_words = current
        st.rerun()


if hasattr(st, "dialog"):
    @st.dialog("Kelola Stop Words")
    def open_stopwords_modal():
        show_stopwords_manager()
        st.button("Tutup", on_click=st.rerun)
else:
    def open_stopwords_modal():
        pass


# ============================================================
#                      SIDEBAR
# ============================================================
with st.sidebar:
    st.title("⚙️ Pengaturan")
    theme_mode = st.radio("Tema Tampilan", ["Light", "Dark"], horizontal=True)
    plotly_template = inject_custom_css(theme_mode)

    st.divider()
    st.subheader("🌐 Bahasa & Teks")
    language = st.selectbox("Bahasa Teks", ["Indonesia", "Inggris"])
    text_type = st.selectbox("Tipe Teks", ["Umum", "Ulasan Produk", "Berita/Artikel", "Media Sosial", "Akademik"])

    st.divider()
    st.subheader("🤖 Model AI")
    num_clusters_input = st.slider("Jumlah Topik (Klaster)", 2, 10, 5)

    st.divider()
    st.subheader("🔧 Preprocessing")
    check_sw = st.checkbox("Hapus Stop Words", value=True, help="Menghapus kata umum yang minim makna.")
    check_lemma = st.checkbox("Aktifkan Lemmatization", value=True, help="Mengubah kata berimbuhan menjadi kata dasar.")
    check_lower = st.checkbox("Case Folding (lowercase)", value=True, help="Mengubah semua teks ke huruf kecil.")
    check_url = st.checkbox("Hapus URL & Mention", value=True, help="Membersihkan URL, mention (@), dan hashtag.")

    if hasattr(st, "dialog"):
        if st.button("📝 Kelola Stop Words", use_container_width=True):
            open_stopwords_modal()
    else:
        with st.expander("📝 Kelola Stop Words"):
            show_stopwords_manager()

    st.markdown("---")
    st.markdown(
        f'<div class="footer-text">Developed by <b>Suwarno</b><br>Powered by <b>OpenAI GPT-4o</b>',
        unsafe_allow_html=True
    )


# ============================================================
#                     MAIN UI
# ============================================================
render_elegant_header(theme_mode)

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = ""
client = OpenAI(api_key=api_key) if api_key else None
MODEL_NAME = "gpt-4o"

# --- INPUT AREA ---
container_input = st.container()
with container_input:
    tab_upload, tab_text = st.tabs(["📂 Unggah Dokumen", "✍️ Teks Langsung"])
    input_text_list = []

    with tab_upload:
        st.info("📁 Format yang didukung: **.csv**, **.xlsx**, **.txt** — Maks. 10 MB")
        uploaded_file = st.file_uploader("Upload File", type=['csv', 'xlsx', 'txt'], label_visibility="collapsed")
        if uploaded_file:
            if uploaded_file.size > 10 * 1024 * 1024:
                st.error("⚠️ File terlalu besar (>10MB). Silakan kurangi ukuran file.")
            else:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        try:
                            df_u = pd.read_csv(uploaded_file, encoding='utf-8')
                        except UnicodeDecodeError:
                            uploaded_file.seek(0)
                            df_u = pd.read_csv(uploaded_file, encoding='latin-1')
                    elif uploaded_file.name.endswith('.xlsx'):
                        df_u = pd.read_excel(uploaded_file)
                    else:
                        try:
                            txt = uploaded_file.read().decode('utf-8')
                        except UnicodeDecodeError:
                            uploaded_file.seek(0)
                            txt = uploaded_file.read().decode('latin-1')
                        df_u = pd.DataFrame(txt.splitlines(), columns=['Teks'])

                    cols = [c for c in df_u.columns if df_u[c].dtype == 'object']
                    if cols:
                        t_col = st.selectbox("Pilih Kolom Teks:", cols)
                        input_text_list = df_u[t_col].dropna().astype(str).tolist()
                        st.success(f"✅ Berhasil dimuat: **{len(input_text_list)}** baris data dari kolom `{t_col}`")
                    else:
                        st.error("Tidak ada kolom bertipe teks.")
                except Exception as e:
                    st.error(f"Gagal membaca file: {str(e)}")

    with tab_text:
        dt = st.text_area("Tempel teks di sini (pisahkan dengan baris baru)...", height=150)
        if dt:
            input_text_list = [t for t in dt.split('\n') if t.strip()]
            if input_text_list:
                st.success(f"✅ **{len(input_text_list)}** baris teks terdeteksi.")


# ============================================================
#                     PROCESSING
# ============================================================
if st.button("🚀 Lakukan Analisis", type="primary", use_container_width=True):
    if not input_text_list:
        st.warning("⚠️ Data kosong. Silakan unggah file atau masukkan teks terlebih dahulu.")
    elif not client:
        st.error("🔑 API Key OpenAI belum dikonfigurasi. Tambahkan di Streamlit Secrets.")
    else:
        with st.spinner("🔄 Sedang memproses analisis lengkap..."):
            df = pd.DataFrame(input_text_list, columns=['Teks_Asli'])

            # --- Preprocessing ---
            status_text = st.empty()
            status_text.info("⏳ Tahap 1/4: Preprocessing teks...")

            factory = StemmerFactory()
            stemmer = factory.create_stemmer() if (language == "Indonesia" and check_lemma) else None

            clean_res = []
            pb = st.progress(0)
            for i, t in enumerate(df['Teks_Asli']):
                clean_res.append(clean_text(t, check_sw, check_lemma, check_lower, st.session_state.stop_words, stemmer))
                if i % 10 == 0 or i == len(df) - 1:
                    pb.progress((i + 1) / len(df))
            pb.empty()

            df['Teks_Clean'] = clean_res
            df = df[df['Teks_Clean'].str.strip() != ""].reset_index(drop=True)

            if len(df) < 2:
                st.error("Data terlalu sedikit setelah preprocessing. Minimal 2 dokumen diperlukan.")
                st.stop()

            # --- Clustering ---
            status_text.info("⏳ Tahap 2/4: Clustering topik dengan TF-IDF + K-Means...")
            k = min(num_clusters_input, len(df))
            if k < 2:
                k = 2

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
                label = get_topic_name_ai(client, MODEL_NAME, top_w[:5]) if top_w else f"Topik {i + 1}"
                topic_map[i] = label
                topic_data_list.append({
                    'Nomor': i + 1,
                    'Topik': label,
                    'Keywords': ", ".join(top_w),
                    'Jumlah_Dokumen': int((df['Cluster_ID'] == i).sum())
                })

            df['Topik'] = df['Cluster_ID'].map(topic_map)

            # --- Sentiment AI ---
            status_text.info("⏳ Tahap 3/4: Analisis sentimen dengan AI...")
            df['Sentimen'] = get_sentiment_ai(client, MODEL_NAME, df['Teks_Asli'].tolist())

            # --- NER Analysis ---
            status_text.info("⏳ Tahap 4/4: Deteksi entitas (NER)...")
            full_text_sample = " ".join(df['Teks_Asli'].tolist())
            ner_result = get_ner_ai(client, MODEL_NAME, full_text_sample)

            status_text.empty()

            # --- Save to state ---
            st.session_state.data = df
            st.session_state.topic_details = topic_data_list
            st.session_state.ner_results = ner_result
            st.session_state.vectorizer = vec
            st.session_state.tfidf_matrix = tfidf
            st.session_state.kmeans = kmeans
            st.session_state.topic_map = topic_map
            st.session_state.analysis_done = True
            st.session_state.summary_cache = None
            st.rerun()


# ============================================================
#                     DASHBOARD
# ============================================================
if st.session_state.analysis_done and st.session_state.data is not None:
    df = st.session_state.data
    st.write("---")

    # --- HEADER METRICS ---
    sc = df['Sentimen'].value_counts().to_dict()
    total_docs = len(df)
    dominant_sentiment = max(sc, key=sc.get) if sc else "-"
    num_topics = df['Topik'].nunique()
    avg_words = round(df['Teks_Clean'].str.split().str.len().mean(), 1)

    # Custom styled metrics
    card_bg = "#1a1d24" if theme_mode == 'Dark' else "#ffffff"
    border_col = "#2d3139" if theme_mode == 'Dark' else "#e2e8f0"
    text_col = "#e8e8e8" if theme_mode == 'Dark' else "#1a1a2e"
    sub_col = "#9ca3af" if theme_mode == 'Dark' else "#64748b"

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="stat-card" style="border-top: 3px solid #4facfe;">
            <div class="stat-value" style="color: #4facfe;">📄 {total_docs}</div>
            <div class="stat-label">Total Dokumen</div></div>""", unsafe_allow_html=True)
    with m2:
        sent_color = SENTIMENT_COLORS.get(dominant_sentiment, '#6c757d')
        st.markdown(f"""<div class="stat-card" style="border-top: 3px solid {sent_color};">
            <div class="stat-value" style="color: {sent_color};">🎭 {dominant_sentiment}</div>
            <div class="stat-label">Sentimen Dominan</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="stat-card" style="border-top: 3px solid #BB8FCE;">
            <div class="stat-value" style="color: #BB8FCE;">📂 {num_topics}</div>
            <div class="stat-label">Topik Terdeteksi</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="stat-card" style="border-top: 3px solid #82E0AA;">
            <div class="stat-value" style="color: #82E0AA;">📏 {avg_words}</div>
            <div class="stat-label">Rata-rata Kata/Dok</div></div>""", unsafe_allow_html=True)

    st.write("")

    # --- TABS ---
    t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
        "📝 Ringkasan AI",
        "🎭 Sentimen",
        "📂 Topik",
        "🔀 Sentimen × Topik",
        "🔠 Kata Kunci",
        "🔗 N-Gram",
        "👤 Entitas (NER)",
        "🌐 Network"
    ])

    # ========================
    # TAB 1: RINGKASAN AI
    # ========================
    with t1:
        st.markdown("### 📝 Laporan Analisis Eksekutif")
        st.caption("AI akan menganalisis seluruh temuan dan menghasilkan laporan terstruktur yang komprehensif.")

        if st.session_state.summary_cache:
            st.markdown(st.session_state.summary_cache)
            if st.button("🔄 Regenerasi Ringkasan"):
                st.session_state.summary_cache = None
                st.rerun()
        else:
            if st.button("✨ Generate Laporan Komprehensif", type="primary"):
                with st.spinner("🤖 AI sedang menyusun laporan eksekutif..."):
                    summary = generate_ai_summary(
                        client, MODEL_NAME, df,
                        sc, st.session_state.topic_details,
                        st.session_state.ner_results, text_type, language
                    )
                    st.session_state.summary_cache = summary
                    st.rerun()

    # ========================
    # TAB 2: SENTIMEN
    # ========================
    with t2:
        st.markdown("### 🎭 Analisis Sentimen")
        c1, c2 = st.columns([1, 2])
        with c1:
            fig_pie = px.pie(
                values=list(sc.values()), names=list(sc.keys()),
                hole=0.45, color=list(sc.keys()),
                color_discrete_map=SENTIMENT_COLORS,
                template=plotly_template
            )
            fig_pie.update_traces(textinfo='percent+label', textfont_size=13)
            fig_pie.update_layout(
                showlegend=False, height=350,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Sentiment counts bar
            df_sc = pd.DataFrame({'Sentimen': sc.keys(), 'Jumlah': sc.values()})
            fig_bar_s = px.bar(
                df_sc, x='Sentimen', y='Jumlah', color='Sentimen',
                color_discrete_map=SENTIMENT_COLORS, template=plotly_template,
                text='Jumlah'
            )
            fig_bar_s.update_traces(textposition='outside')
            fig_bar_s.update_layout(showlegend=False, height=250, margin=dict(t=10, b=10))
            st.plotly_chart(fig_bar_s, use_container_width=True)

        with c2:
            f_s = st.multiselect("Filter Sentimen:", df['Sentimen'].unique(), default=list(df['Sentimen'].unique()))

            def color_sentiment(v):
                if v == 'Positif':
                    return 'background-color: #28a745; color: white'
                if v == 'Negatif':
                    return 'background-color: #dc3545; color: white'
                if v == 'Netral':
                    return 'background-color: #ffc107; color: black'
                return ''

            filtered = df[df['Sentimen'].isin(f_s)][['Teks_Asli', 'Topik', 'Sentimen']]
            st.dataframe(
                filtered.style.map(color_sentiment, subset=['Sentimen']),
                use_container_width=True, height=500
            )

    # ========================
    # TAB 3: TOPIK
    # ========================
    with t3:
        st.markdown("### 📂 Analisis Topik (Clustering)")

        tc = df['Topik'].value_counts().reset_index()
        tc.columns = ['Topik', 'Jumlah']
        fig_bar_t = px.bar(
            tc, x='Jumlah', y='Topik', orientation='h',
            color='Jumlah', color_continuous_scale='Blues',
            template=plotly_template, text='Jumlah'
        )
        fig_bar_t.update_traces(textposition='outside')
        fig_bar_t.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
        st.plotly_chart(fig_bar_t, use_container_width=True)

        # PCA Scatter
        if hasattr(st.session_state, 'tfidf_matrix') and hasattr(st.session_state, 'topic_map'):
            fig_scatter = generate_cluster_scatter(
                st.session_state.tfidf_matrix,
                st.session_state.kmeans.labels_,
                st.session_state.topic_map,
                plotly_template
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("#### 📋 Detail Kata Kunci per Topik")
        df_topics = pd.DataFrame(st.session_state.topic_details)
        st.dataframe(df_topics, hide_index=True, use_container_width=True, column_config={
            "Nomor": st.column_config.NumberColumn("No.", width="small"),
            "Topik": st.column_config.TextColumn("Nama Topik", width="medium"),
            "Keywords": st.column_config.TextColumn("Kata Kunci (Top 10)", width="large"),
            "Jumlah_Dokumen": st.column_config.NumberColumn("Jumlah Dok.", width="small")
        })

    # ========================
    # TAB 4: CROSS ANALYSIS (NEW)
    # ========================
    with t4:
        st.markdown("### 🔀 Cross-Analysis: Sentimen × Topik")
        st.caption("Melihat distribusi sentimen di dalam setiap topik untuk menemukan pola dan area bermasalah.")

        cross_tab = pd.crosstab(df['Topik'], df['Sentimen'])
        for col in ['Positif', 'Negatif', 'Netral']:
            if col not in cross_tab.columns:
                cross_tab[col] = 0
        cross_tab = cross_tab[['Positif', 'Netral', 'Negatif']]

        # Stacked bar
        fig_stack = go.Figure()
        for sent, color in SENTIMENT_COLORS.items():
            if sent in cross_tab.columns:
                fig_stack.add_trace(go.Bar(
                    name=sent, y=cross_tab.index, x=cross_tab[sent],
                    orientation='h', marker_color=color, text=cross_tab[sent],
                    textposition='inside'
                ))
        fig_stack.update_layout(
            barmode='stack', template=plotly_template,
            yaxis={'categoryorder': 'total ascending'},
            height=max(300, len(cross_tab) * 50 + 100),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig_stack, use_container_width=True)

        # Heatmap
        cross_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0).round(2) * 100
        fig_heat = px.imshow(
            cross_pct, text_auto='.0f',
            color_continuous_scale='RdYlGn',
            labels=dict(x="Sentimen", y="Topik", color="Persen (%)"),
            template=plotly_template,
            aspect='auto'
        )
        fig_heat.update_layout(height=max(300, len(cross_tab) * 45 + 100))
        st.plotly_chart(fig_heat, use_container_width=True)

        # Insight tabel
        st.markdown("#### 📊 Tabel Cross-Tabulation")
        cross_display = cross_tab.copy()
        cross_display['Total'] = cross_display.sum(axis=1)
        cross_display['% Positif'] = (cross_display['Positif'] / cross_display['Total'] * 100).round(1)
        cross_display['% Negatif'] = (cross_display['Negatif'] / cross_display['Total'] * 100).round(1)
        st.dataframe(cross_display, use_container_width=True)

    # ========================
    # TAB 5: KATA KUNCI
    # ========================
    with t5:
        st.markdown("### 🔠 Analisis Kata Kunci")
        txt_all = " ".join(df['Teks_Clean'])
        if txt_all.strip():
            wc_bg = 'black' if theme_mode == 'Dark' else 'white'
            wc = WordCloud(
                width=800, height=400, background_color=wc_bg,
                colormap='viridis', max_words=100,
                contour_width=1, contour_color=wc_bg
            ).generate(txt_all)
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5), facecolor=wc_bg)
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis("off")
            st.pyplot(fig_wc)
            plt.close(fig_wc)

        # Top TF-IDF
        sum_tfidf = st.session_state.tfidf_matrix.sum(axis=0)
        words = [(word, sum_tfidf[0, idx]) for word, idx in st.session_state.vectorizer.vocabulary_.items()]
        words = sorted(words, key=lambda x: x[1], reverse=True)[:15]
        df_k = pd.DataFrame(words, columns=["Kata", "Skor TF-IDF"])
        fig_k = px.bar(
            df_k, x="Skor TF-IDF", y="Kata", orientation='h',
            template=plotly_template, color="Skor TF-IDF",
            color_continuous_scale="Blues", text=df_k['Skor TF-IDF'].round(2)
        )
        fig_k.update_traces(textposition='outside')
        fig_k.update_layout(yaxis={'categoryorder': 'total ascending'}, height=450)
        st.plotly_chart(fig_k, use_container_width=True)

        # Word frequency (raw count)
        with st.expander("📊 Frekuensi Kata Mentah (Raw Count)"):
            all_words = txt_all.split()
            wf = Counter(all_words).most_common(20)
            df_wf = pd.DataFrame(wf, columns=["Kata", "Frekuensi"])
            fig_wf = px.bar(df_wf, x="Frekuensi", y="Kata", orientation='h', template=plotly_template)
            fig_wf.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
            st.plotly_chart(fig_wf, use_container_width=True)

    # ========================
    # TAB 6: N-GRAM
    # ========================
    with t6:
        st.markdown("### 🔗 Analisis Frasa (N-Gram)")
        st.caption("Mendeteksi frasa yang sering muncul bersamaan dalam teks.")

        ng1, ng2 = st.columns(2)
        with ng1:
            st.markdown("**Bigram (2 Kata)**")
            bigrams = get_ngrams(df['Teks_Clean'], n=2, top_k=10)
            if bigrams:
                df_bi = pd.DataFrame(bigrams, columns=["Frasa", "Frekuensi"])
                fig_bi = px.bar(
                    df_bi, x="Frekuensi", y="Frasa", orientation='h',
                    template=plotly_template, color="Frekuensi",
                    color_continuous_scale="Teal", text="Frekuensi"
                )
                fig_bi.update_traces(textposition='outside')
                fig_bi.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                st.plotly_chart(fig_bi, use_container_width=True)
            else:
                st.info("Data tidak cukup untuk analisis Bigram.")

        with ng2:
            st.markdown("**Trigram (3 Kata)**")
            trigrams = get_ngrams(df['Teks_Clean'], n=3, top_k=10)
            if trigrams:
                df_tri = pd.DataFrame(trigrams, columns=["Frasa", "Frekuensi"])
                fig_tri = px.bar(
                    df_tri, x="Frekuensi", y="Frasa", orientation='h',
                    template=plotly_template, color="Frekuensi",
                    color_continuous_scale="Purp", text="Frekuensi"
                )
                fig_tri.update_traces(textposition='outside')
                fig_tri.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                st.plotly_chart(fig_tri, use_container_width=True)
            else:
                st.info("Data tidak cukup untuk analisis Trigram.")

        # Fourgram (new)
        with st.expander("🔬 Lihat Fourgram (4 Kata)"):
            fourgrams = get_ngrams(df['Teks_Clean'], n=4, top_k=10)
            if fourgrams:
                df_fg = pd.DataFrame(fourgrams, columns=["Frasa", "Frekuensi"])
                fig_fg = px.bar(
                    df_fg, x="Frekuensi", y="Frasa", orientation='h',
                    template=plotly_template, color="Frekuensi", text="Frekuensi"
                )
                fig_fg.update_traces(textposition='outside')
                fig_fg.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                st.plotly_chart(fig_fg, use_container_width=True)
            else:
                st.info("Data tidak cukup untuk analisis Fourgram.")

    # ========================
    # TAB 7: NER
    # ========================
    with t7:
        st.markdown("### 👤 Named Entity Recognition (NER)")
        st.caption("Entitas penting yang dideteksi secara otomatis oleh AI dari keseluruhan dokumen.")

        if st.session_state.ner_results:
            res = st.session_state.ner_results

            if res.get("Error"):
                st.error(f"Terjadi error pada NER: {res['Error']}")

            col_p, col_o, col_l = st.columns(3)

            with col_p:
                persons = res.get("Person", [])
                st.markdown(f"""<div class="ner-card" style="border-left-color: #4ECDC4;">
                    <h4 style="margin: 0 0 8px 0;">🧑 Person ({len(persons)})</h4>
                </div>""", unsafe_allow_html=True)
                if persons:
                    for p in persons:
                        st.markdown(f"- **{p}**")
                else:
                    st.write("Tidak ditemukan")

            with col_o:
                orgs = res.get("Organization", [])
                st.markdown(f"""<div class="ner-card" style="border-left-color: #45B7D1;">
                    <h4 style="margin: 0 0 8px 0;">🏢 Organization ({len(orgs)})</h4>
                </div>""", unsafe_allow_html=True)
                if orgs:
                    for o in orgs:
                        st.markdown(f"- **{o}**")
                else:
                    st.write("Tidak ditemukan")

            with col_l:
                locs = res.get("Location", [])
                st.markdown(f"""<div class="ner-card" style="border-left-color: #FFA07A;">
                    <h4 style="margin: 0 0 8px 0;">📍 Location ({len(locs)})</h4>
                </div>""", unsafe_allow_html=True)
                if locs:
                    for loc in locs:
                        st.markdown(f"- **{loc}**")
                else:
                    st.write("Tidak ditemukan")

            # NER Summary Chart
            ner_counts = {
                'Person': len(persons),
                'Organization': len(orgs),
                'Location': len(locs)
            }
            if sum(ner_counts.values()) > 0:
                st.markdown("---")
                st.markdown("#### 📊 Distribusi Entitas")
                fig_ner = px.bar(
                    x=list(ner_counts.keys()), y=list(ner_counts.values()),
                    color=list(ner_counts.keys()),
                    color_discrete_sequence=['#4ECDC4', '#45B7D1', '#FFA07A'],
                    template=plotly_template,
                    labels={'x': 'Tipe Entitas', 'y': 'Jumlah'},
                    text=list(ner_counts.values())
                )
                fig_ner.update_traces(textposition='outside')
                fig_ner.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig_ner, use_container_width=True)
        else:
            st.error("Gagal memuat hasil NER.")

    # ========================
    # TAB 8: NETWORK
    # ========================
    with t8:
        st.markdown("### 🌐 Text Network Analysis")
        st.caption("Visualisasi hubungan antara topik (node besar) dan kata kunci dominan (node kecil).")

        if st.session_state.topic_details:
            net_mode = st.radio(
                "Pilih mode visualisasi:",
                ["Interaktif (Plotly)", "Statis (Matplotlib)"],
                horizontal=True
            )
            if net_mode == "Interaktif (Plotly)":
                fig_net = generate_text_network_plotly(st.session_state.topic_details, theme_mode)
                st.plotly_chart(fig_net, use_container_width=True)
            else:
                fig_net = generate_text_network(st.session_state.topic_details, theme_mode)
                st.pyplot(fig_net)
                plt.close()
        else:
            st.warning("Data topik belum tersedia.")

    # ========================
    # EXPORT SECTION
    # ========================
    st.divider()
    st.markdown("### 📥 Ekspor Data")

    exp1, exp2, exp3 = st.columns(3)

    with exp1:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Unduh CSV (Data Lengkap)",
            csv_data, "anatext_analysis.csv", "text/csv",
            use_container_width=True
        )

    with exp2:
        # Export topic summary
        df_topic_export = pd.DataFrame(st.session_state.topic_details)
        st.download_button(
            "📥 Unduh Topik & Keywords",
            df_topic_export.to_csv(index=False).encode('utf-8'),
            "anatext_topics.csv", "text/csv",
            use_container_width=True
        )

    with exp3:
        # Export summary report as text
        if st.session_state.summary_cache:
            st.download_button(
                "📥 Unduh Laporan AI (TXT)",
                st.session_state.summary_cache.encode('utf-8'),
                "anatext_report.txt", "text/plain",
                use_container_width=True
            )
        else:
            st.button("📥 Generate laporan dulu", disabled=True, use_container_width=True)


