import numpy as np
import requests
import streamlit as st
import chromadb
import ollama
from pypdf import PdfReader
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Agriculture Assistant", page_icon="🌱", layout="wide")
LLM = "llama3.2:3b"        # change if you use another Ollama model
EMB = "nomic-embed-text"
DISCLAIMER = "⚠️ AI-based information. Consult a local agriculture officer before using any fertilizer or pesticide."

# ---------------- Telugu support (translate in / out) ----------------
def translate(text, target):
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="auto", target=target).translate(text)
    except Exception:
        return text

lang = st.sidebar.radio("Language / భాష", ["English", "తెలుగు"])
st.sidebar.caption("Telugu uses online translation around the English RAG pipeline.")

# ---------------- RAG: PDF -> chunks -> embeddings -> ChromaDB ----------------
def embed(text):
    return ollama.embeddings(model=EMB, prompt=text)["embedding"]

def build_index(files):
    client = chromadb.Client()
    try:
        client.delete_collection("docs")
    except Exception:
        pass
    col = client.create_collection("docs")
    for f in files:
        for pno, page in enumerate(PdfReader(f).pages, 1):
            text = page.extract_text() or ""
            for i in range(0, len(text), 700):          # 900-char chunks, 200 overlap
                chunk = text[i:i + 900].strip()
                if len(chunk) > 50:
                    col.add(ids=[f"{f.name}-{pno}-{i}"], documents=[chunk],
                            embeddings=[embed(chunk)],
                            metadatas=[{"src": f"{f.name} (page {pno})"}])
    return col

def answer(col, question):
    q_en = translate(question, "en") if lang != "English" else question
    res = col.query(query_embeddings=[embed(q_en)], n_results=3)
    context = "\n---\n".join(res["documents"][0])
    sources = sorted({m["src"] for m in res["metadatas"][0]})
    prompt = ("Answer ONLY using the context below. If the answer is not in the context, "
              "say 'Not found in the documents.' Do not give exact pesticide/fertilizer "
              f"doses unless the context states them.\n\nContext:\n{context}\n\nQuestion: {q_en}")
    reply = ollama.chat(model=LLM, messages=[{"role": "user", "content": prompt}])["message"]["content"]
    if lang != "English":
        reply = translate(reply, "te")
    return reply, sources

# ---------------- ML: crop recommendation (Random Forest) ----------------
# Typical ideal conditions: N, P, K, temp C, humidity %, pH, rainfall mm
CROPS = {
    "Rice":     [80, 48, 40, 23, 82, 6.4, 236],
    "Maize":    [78, 48, 20, 22, 65, 6.2, 85],
    "Cotton":   [118, 46, 20, 24, 80, 6.9, 80],
    "Chickpea": [40, 68, 80, 18, 17, 7.3, 80],
    "Banana":   [100, 82, 50, 27, 80, 6.0, 105],
    "Mango":    [20, 27, 30, 31, 50, 5.8, 95],
}

@st.cache_resource
def train_model():
    rng = np.random.default_rng(42)
    X, y = [], []
    for crop, c in CROPS.items():
        c = np.array(c, dtype=float)
        X.append(rng.normal(c, c * 0.10, size=(200, 7)))   # synthetic samples around ideal values
        y += [crop] * 200
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(np.vstack(X), y)
    return model

# ---------------- Weather: Open-Meteo (live, no API key) ----------------
def get_weather(city):
    g = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": city, "count": 1}, timeout=10).json()
    if "results" not in g:
        return None
    r = g["results"][0]
    w = requests.get("https://api.open-meteo.com/v1/forecast", timeout=10, params={
        "latitude": r["latitude"], "longitude": r["longitude"],
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "daily": "precipitation_sum", "forecast_days": 3}).json()
    return r["name"], w

# ---------------- UI ----------------
st.title("🌱 AI Agriculture Assistant")
tabs = st.tabs(["📚 Ask Documents", "🌾 Crop Recommendation", "🌦️ Weather Advice",
                "🧪 Soil Health", "🧮 Expense Calculator"])

# 1. Document chatbot
with tabs[0]:
    files = st.file_uploader("Upload agriculture PDF(s)", type="pdf", accept_multiple_files=True)
    if files and st.button("Index documents"):
        with st.spinner("Reading PDFs and creating embeddings..."):
            st.session_state["col"] = build_index(files)
        st.success("Ready! Ask a question below.")
    q = st.text_input("Ask your question" if lang == "English" else "మీ ప్రశ్న అడగండి")
    if q:
        if "col" not in st.session_state:
            st.warning("Upload and index a PDF first.")
        else:
            with st.spinner("Thinking..."):
                reply, srcs = answer(st.session_state["col"], q)
            st.write(reply)
            st.caption("Sources: " + ", ".join(srcs))
            st.info(DISCLAIMER)

# 2. Crop recommendation
with tabs[1]:
    st.subheader("Enter soil and weather values")
    c1, c2, c3 = st.columns(3)
    n = c1.number_input("Nitrogen", 0, 200, 80)
    p = c2.number_input("Phosphorus", 0, 200, 45)
    k = c3.number_input("Potassium", 0, 200, 40)
    t = c1.number_input("Temperature (°C)", 0.0, 50.0, 24.0)
    h = c2.number_input("Humidity (%)", 0.0, 100.0, 70.0)
    ph = c3.number_input("pH", 0.0, 14.0, 6.5)
    rain = c1.number_input("Rainfall (mm)", 0.0, 300.0, 120.0)
    if st.button("Recommend crop"):
        model = train_model()
        probs = model.predict_proba([[n, p, k, t, h, ph, rain]])[0]
        top = np.argsort(probs)[::-1][:3]
        for i in top:
            st.write(f"**{model.classes_[i]}**: {probs[i]*100:.0f}% match")
            st.progress(float(probs[i]))
        st.caption("Model: Random Forest trained on synthetic data around typical crop conditions. "
                   "Replace with the Kaggle Crop Recommendation CSV for real training.")

# 3. Weather advice (API gives facts, Python rules give advice)
with tabs[2]:
    city = st.text_input("City", "Hyderabad")
    crop = st.selectbox("Crop", list(CROPS.keys()))
    if st.button("Get advice"):
        try:
            data = get_weather(city)
            if not data:
                st.error("City not found.")
            else:
                name, w = data
                cur = w["current"]
                rain3 = sum(w["daily"]["precipitation_sum"])
                st.metric(f"{name} temperature", f"{cur['temperature_2m']} °C")
                st.write(f"Humidity: {cur['relative_humidity_2m']}% | Wind: {cur['wind_speed_10m']} km/h "
                         f"| Rain (next 3 days): {rain3:.1f} mm")
                tips = []
                if rain3 > 10: tips.append("Heavy rain expected: postpone irrigation and spraying, and ensure drainage.")
                elif rain3 < 1: tips.append("Little rain expected: plan irrigation.")
                if cur["temperature_2m"] > 35: tips.append("High heat: irrigate in early morning or evening.")
                if cur["relative_humidity_2m"] > 80: tips.append("High humidity: watch for fungal diseases.")
                if cur["wind_speed_10m"] > 25: tips.append("Strong wind: avoid pesticide spraying.")
                for tip in tips or ["Conditions look normal."]:
                    st.success(f"{crop}: " + (translate(tip, "te") if lang != "English" else tip))
        except Exception as e:
            st.error(f"Weather service unavailable: {e}")

# 4. Soil health (rule-based)
with tabs[3]:
    sph = st.number_input("Soil pH", 0.0, 14.0, 6.5, key="sph")
    sn = st.number_input("Nitrogen (kg/ha)", 0, 600, 250, key="sn")
    sp = st.number_input("Phosphorus (kg/ha)", 0, 100, 20, key="sp")
    sk = st.number_input("Potassium (kg/ha)", 0, 600, 200, key="sk")
    if st.button("Analyse soil"):
        st.write("**pH:** " + ("Acidic: consider liming after a soil test." if sph < 6
                               else "Alkaline: consider organic matter/gypsum." if sph > 7.5
                               else "Good range for most crops."))
        st.write("**Nitrogen:** " + ("Low" if sn < 280 else "Medium" if sn < 560 else "High"))
        st.write("**Phosphorus:** " + ("Low" if sp < 10 else "Medium" if sp < 25 else "High"))
        st.write("**Potassium:** " + ("Low" if sk < 108 else "Medium" if sk < 280 else "High"))
        st.info("Get a lab soil test before deciding fertilizer quantities.")

# 5. Expense calculator (plain Python, no AI needed)
with tabs[4]:
    cols = st.columns(3)
    items = ["Seeds", "Fertilizer", "Pesticides", "Labour", "Irrigation", "Other"]
    costs = [cols[i % 3].number_input(f"{x} (₹)", 0.0, key=f"c{i}") for i, x in enumerate(items)]
    revenue = st.number_input("Expected revenue (₹)", 0.0)
    total = sum(costs)
    st.metric("Total cost", f"₹{total:,.0f}")
    st.metric("Estimated profit", f"₹{revenue - total:,.0f}")
    if total > 0:
        st.bar_chart(dict(zip(items, costs)))
