# app/eval/push_dataset.py
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from langsmith.utils import LangSmithError

load_dotenv()

# ---- In-memory examples (your list) ----
examples = [
    {"inputs": {"question": "Can you walk me through your academic path and how it shaped your move into data science and AI?"},
     "outputs": {"answer": "I began in neuroscience, analyzing EEG data, which introduced me to signal processing and statistical modeling. Those skills naturally led into machine learning applications, and over time I shifted toward applied data science and AI. My academic path gave me a rigorous foundation in quantitative thinking that I carry into industry work."}},
    {"inputs": {"question": "How did your EEG and neuroscience research prepare you for applied ML work?"},
     "outputs": {"answer": "Working with EEG required cleaning noisy signals and extracting meaningful features, which is directly analogous to feature engineering in ML. It also trained me to think critically about experimental design and evaluation. That experience translates well into designing robust ML pipelines."}},
    {"inputs": {"question": "Tell me about a project where you combined signal processing techniques with machine learning."},
     "outputs": {"answer": "In my postdoc, I applied time-frequency analyses to EEG data and then trained classifiers to predict cognitive states. The project combined Fourier transforms with supervised learning to link neural signals to behavior. It showed me how domain knowledge and ML can reinforce each other."}},
    {"inputs": {"question": "You’ve published scientific papers — how has that experience influenced how you communicate technical results today?"},
     "outputs": {"answer": "Publishing trained me to be precise, concise, and transparent about methods and limitations. I bring that same approach when communicating results to technical and non-technical audiences in industry. It helps me frame findings clearly and build trust with stakeholders."}},
    {"inputs": {"question": "Describe one of the projects in your CV that best demonstrates your ability to connect research to business impact."},
     "outputs": {"answer": "I built a recommendation model for an online platform that improved engagement metrics significantly. The project moved from prototyping to production and showed measurable business gains. It reflects my ability to translate research into applied value."}},
    {"inputs": {"question": "What was the most challenging project you led, and how did you overcome obstacles to deliver results?"},
     "outputs": {"answer": "A complex ML pipeline was underperforming due to noisy input data and latency constraints. I re-architected the preprocessing, added caching, and simplified the model. These changes balanced accuracy with efficiency and allowed us to deploy successfully."}},
    {"inputs": {"question": "How have you used embeddings in real projects?"},
     "outputs": {"answer": "I’ve used text embeddings to build semantic search and retrieval systems, including a RAG app that grounded answers in my own documents. Embeddings helped link unstructured text to meaningful outputs. They were also valuable for clustering and topic discovery."}},
    {"inputs": {"question": "Can you share an example of building or improving a retrieval-augmented generation (RAG) system?"},
     "outputs": {"answer": "I built a RAG application that indexed my resume, CV, and project documents with FAISS. The system retrieved context snippets and used them to generate accurate, first-person interview answers. This improved relevance and ensured answers stayed grounded in real material."}},
    {"inputs": {"question": "Walk me through how you designed evaluations for an ML system you’ve worked on."},
     "outputs": {"answer": "For one model, I created a golden dataset of expected Q&A pairs to evaluate precision and recall. I also included user-centered metrics like helpfulness and latency. This evaluation framework helped iterate quickly and track progress over time."}},
    {"inputs": {"question": "How do you ensure models you build are both technically sound and aligned with business needs?"},
     "outputs": {"answer": "I start by clarifying the business objectives and success metrics. Then I design experiments and evaluations that align with those outcomes. This ensures the model is not only accurate but also impactful for the business."}},
    {"inputs": {"question": "Tell me about a time when you had to simplify technical findings for a non-technical audience."},
     "outputs": {"answer": "I once explained a model’s false positives using an analogy to weather forecasts. By focusing on the concept of trade-offs, the stakeholders understood the key point without needing technical detail. It built alignment around decision-making."}},
    {"inputs": {"question": "Which of your projects do you consider the most innovative, and why?"},
     "outputs": {"answer": "The RAG-based interview assistant was highly innovative because it combined personal documents, retrieval, and LLMs into a functional agent. It demonstrated how to tailor AI for individual use cases. The novelty was in making it both accurate and personal."}},
    {"inputs": {"question": "How have you incorporated external research (like published papers) into your own applied work?"},
     "outputs": {"answer": "I regularly consult papers for algorithms or evaluation methods. For example, I integrated recent approaches to embeddings into a semantic search pipeline. Using academic research ensures I’m applying state-of-the-art techniques responsibly."}},
    {"inputs": {"question": "Tell me about a time you improved an ML pipeline or tool that wasn’t working as expected."},
     "outputs": {"answer": "I diagnosed bottlenecks in a model deployment pipeline that caused unacceptable latency. By profiling steps and optimizing preprocessing, I reduced response times significantly. The improvements made the pipeline viable for production use."}},
    {"inputs": {"question": "What project in your portfolio best shows your ability to link theory to practice?"},
     "outputs": {"answer": "My work with EEG showed theoretical grounding in neuroscience and signal processing, but I extended it by building ML models that predicted cognitive states. That bridge from theory to practice has guided many of my applied projects since."}},
    {"inputs": {"question": "How has your interdisciplinary background (science + data + product) given you an edge in your work?"},
     "outputs": {"answer": "It helps me see problems from multiple angles. I can dive deep into data, but also step back and think about user experience and product impact. That combination lets me build solutions that are both technically rigorous and practical."}},
    {"inputs": {"question": "Describe a project where latency or efficiency was a key requirement."},
     "outputs": {"answer": "In one RAG project, latency was capped at 5–10 seconds. I optimized retrieval with FAISS and cached frequent queries to meet those limits. The work balanced user experience with technical feasibility."}},
    {"inputs": {"question": "Can you share an example where you evaluated trade-offs between model complexity and deployment constraints?"},
     "outputs": {"answer": "I compared a deep model with a simpler logistic regression for a classification task. While the deep model had slightly higher accuracy, it was too slow. I chose the simpler model, which met latency requirements and was easier to maintain."}},
    {"inputs": {"question": "How do you think your prior academic collaborations prepared you for working with cross-functional product teams?"},
     "outputs": {"answer": "Academic work trained me to collaborate with people from different disciplines, from psychologists to engineers. That experience maps directly to product teams where communication and alignment are essential."}},
    {"inputs": {"question": "Tell me about a project you designed that had measurable business impact."},
     "outputs": {"answer": "I developed an ML-driven recommendation pipeline that improved click-through rates significantly. The result was a measurable uptick in revenue. It demonstrated how data science could tie directly to business KPIs."}},
    {"inputs": {"question": "Which personal or research project most excites you, and what does it say about your goals?"},
     "outputs": {"answer": "The interview assistant project excites me because it blends personal expression with cutting-edge AI. It reflects my goal of making AI more human-centered and useful for individual growth."}},
    {"inputs": {"question": "How have you handled situations where the data available wasn’t ideal for the problem?"},
     "outputs": {"answer": "I’ve used data augmentation, proxy variables, or combined multiple sources to compensate. For example, in EEG work I developed preprocessing to reduce noise. The key is to be transparent about limitations while still delivering value."}},
    {"inputs": {"question": "Tell me about a project in your CV where you had to balance experimentation with deadlines."},
     "outputs": {"answer": "I worked on a product launch where timelines were fixed. I ran rapid experiments but locked down the pipeline once performance was good enough. That balance allowed us to ship on time without compromising quality."}},
    {"inputs": {"question": "Have you done any work related to inference priming or cognitive neuroscience methods that you think translates into AI work?"},
     "outputs": {"answer": "Yes, I studied inference priming in EEG experiments. The way context influences interpretation is very similar to how embeddings shape LLM responses. That background informs how I think about context in AI systems."}},
    {"inputs": {"question": "Where do you see your unique background (academic + applied + product) taking you in the AI field over the next five years?"},
     "outputs": {"answer": "I see myself leading applied AI projects that balance cutting-edge research with product value. My interdisciplinary background positions me to bridge teams and ensure innovations are impactful. My goal is to help shape responsible, human-centered AI."}},
]

DATASET_NAME = os.getenv("LS_DATASET", "Interview_Agent_QAS")
DATASET_DESC = "Golden interview Q&A."

# LangSmith client (use correct arg name: base_url)
client = Client(
    api_key=os.getenv("LANGSMITH_API_KEY"),
    api_url=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
)

# --- Prefer pushing the in-memory examples ---
try:
    ds = client.create_dataset(dataset_name=DATASET_NAME, description=DATASET_DESC)
except LangSmithError:
    # If already exists or creation forbidden, try to read it
    ds = client.read_dataset(dataset_name=DATASET_NAME)

client.create_examples(dataset_id=ds["id"], examples=examples)
print(f"Pushed {len(examples)} examples to dataset '{ds['name']}'.")

# --- Optional: upload CSV instead (if you already have one) ---
# To use: export QAS_CSV=/absolute/or/relative/path/to/qas_dataset.csv
csv_env = os.getenv("QAS_CSV")
if csv_env:
    csv_file = Path(csv_env).expanduser().resolve()
    if csv_file.exists():
        uploaded = client.upload_csv(
            csv_file=csv_file,
            input_keys=["question"],
            output_keys=["answer"],
            name=DATASET_NAME,          # uploads into same-named dataset
            description=DATASET_DESC,
            data_type="kv",
        )
        print(f"CSV uploaded into dataset '{uploaded['name']}'.")
    else:
        print(f"[warn] QAS_CSV path does not exist: {csv_file}")
