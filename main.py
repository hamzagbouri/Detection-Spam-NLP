import streamlit as st
import sys
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from custom_transformers import StemmerTransformer 
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header { font-size: 3rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .prediction-box { padding: 1.5rem; border-radius: 10px; margin: 1rem 0; font-size: 1.2rem; text-align: center; }
    .spam-box { background-color: #ffebee; border: 2px solid #f44336; color: #c62828; }
    .ham-box { background-color: #e8f5e9; border: 2px solid #4caf50; color: #2e7d32; }
    .stButton>button { width: 100%; background-color: #1f77b4; color: white; font-size: 1.1rem; padding: 0.5rem 1rem; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

def init_spark():
    return SparkSession.builder \
        .appName("SpamDetectorApp") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true") \
        .config("spark.python.worker.faulthandler.enabled", "true") \
        .getOrCreate()

def load_model(model_path):
    return PipelineModel.load(model_path)

spark = init_spark()
MODEL_PATH = r"C:\Users\hamza\Desktop\Detection-Spam-NLP\models\pipeline_model"
model = None
page = "🔍 Détecteur de Spam"

if page == "🔍 Détecteur de Spam":
    st.markdown('<h1 class="main-header">📧 Détecteur de Spam</h1>', unsafe_allow_html=True)

    if model is None:
        try:
            model = load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"⚠️ Le modèle n'a pas pu être chargé : {e}")
            st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Saisir un email")
        subject = st.text_input("Objet de l'email", placeholder="Ex: Offre spéciale limitée !")
        message = st.text_area("Contenu de l'email", placeholder="Entrez le contenu...", height=200)
        predict_button = st.button("🔍 Analyser", type="primary", use_container_width=True)

    with col2:
        st.subheader("📊 Résultats")

        if predict_button:
            if not message.strip() and not subject.strip():
                st.warning("⚠️ Veuillez saisir au moins un sujet ou un message.")
            else:
                with st.spinner("Analyse en cours..."):
                    text = f"{subject or ''} {message or ''}".strip()
                    df_input = spark.createDataFrame([(text,)], ["text"])
                    try:
                        result = model.transform(df_input).collect()[0]
                        prediction = int(result.prediction)
                        probability = float(result.probability[1]) if hasattr(result, "probability") else None

                        if prediction == 1:
                            st.markdown('<div class="prediction-box spam-box"><h2>🚨 SPAM</h2><p>Cet email a été classé comme spam.</p></div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="prediction-box ham-box"><h2>✅ HAM (Légitime)</h2><p>Cet email semble légitime.</p></div>', unsafe_allow_html=True)

                        if probability is not None:
                            st.metric("Probabilité de spam", f"{probability:.2%}")
                            st.progress(probability)
                            if probability > 0.8:
                                st.info("🔴 Confiance élevée - Très probablement un spam")
                            elif probability > 0.5:
                                st.warning("🟡 Confiance modérée - À examiner avec attention")
                            else:
                                st.success("🟢 Confiance faible - Probablement légitime")

                    except Exception as e:
                        st.error(f"❌ Erreur lors de la prédiction : {e}")
        else:
            st.info("👆 Cliquez sur 'Analyser' pour commencer")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Détecteur de Spam - Application ML</div>", unsafe_allow_html=True)