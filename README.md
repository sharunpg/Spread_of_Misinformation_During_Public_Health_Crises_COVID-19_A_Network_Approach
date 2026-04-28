# 📚 Spread of Misinformation During Public Health Crises COVID-19 : A Network Approach

This project develops an AI-based misinformation detection system that verifies COVID-19 related health claims and classifies them as **CORRECT, MISINFORMATION, or UNVERIFIED**. The system accepts claims from multiple input formats such as **text, URLs, and images**, processes them through a structured verification pipeline, and compares them with a **locally curated knowledge base from trusted sources such as WHO, CDC, and NHS**.

The goal of the system is to **reduce the spread of false health information during public health crises by providing an explainable, privacy-preserving, and offline misinformation detection framework.**

---

# How the System Works (Step-by-Step)

The system follows a **multi-stage verification pipeline**. According to the system architecture diagram in the report (Chapter 5), each module processes the claim sequentially before producing the final classification.

---

## Step 1: User Input

The process begins when the user submits a claim through the **Streamlit web interface**.

The system supports three input types:

### Text Input

Example:  
“Drinking hot water cures COVID-19.”

### URL Input

The system extracts textual content from the webpage.

### Image Input

Example: screenshot of a social media post.

Images are processed using **EasyOCR** to extract the text from the image.

---

## Step 2: Input Processing

All inputs are converted into **standard text format** so that they can be analyzed uniformly.

Processing steps include:

- Extract text from URL pages
- Extract text from images using OCR
- Pass text inputs directly to preprocessing

This ensures that the system can handle **multi-modal information sources**.

---

## Step 3: Text Preprocessing

The system cleans and standardizes the extracted claim before analysis.

Preprocessing includes:

- Language detection using **langdetect**
- Translation of non-English text using **deep-translator**
- Removing punctuation and special characters
- Converting text to lowercase
- Formatting the text

These steps ensure consistent input for further analysis.

---

## Step 4: Knowledge Base Comparison

The system maintains an **offline knowledge base** containing:

- Verified facts
- Known misinformation patterns

These are collected from trusted health organizations such as **WHO, CDC, and NHS**.

Each entry in the knowledge base is converted into **vector embeddings using Sentence-BERT**.

The input claim is also converted into an embedding.

Then the system compares them using:

**Cosine Similarity**

This helps detect **paraphrased or semantically similar misinformation claims**, not just exact keyword matches.

---

## Step 5: Intent Detection

The system analyzes the **type of medical claim** using rule-based intent detection.

Claims are categorized as:

- General wellness
- Symptom management
- Prevention
- Treatment
- Cure

For example:

Claim:  
“Garlic cures COVID-19”

Intent detected → **Cure claim**

The system also performs **negation detection**, identifying phrases such as:

- “does not cure”
- “cannot prevent”

This prevents incorrect classification of corrective statements.

---

## Step 6: Entity Validation

The system extracts important entities from the claim using **spaCy Named Entity Recognition (NER)**.

Examples of entities:

- Vaccine names
- Drug names
- Organizations
- Numerical values

These entities are compared with **structured data from offline Wikidata** to detect factual inconsistencies.

---

## Step 7: Network-Based Risk Analysis

The system also analyzes misinformation patterns using a **similarity network**.

Algorithms used:

- **MinHash**
- **Locality Sensitive Hashing (LSH)**

These algorithms identify claims that are **similar to previously detected misinformation claims**.

The system calculates a **network risk score**:

```
Risk Score =
(Number of misinformation neighbors) / (Total similar claims)
```

If a claim is similar to many misinformation claims, the system increases the **misinformation probability**.

---

## Step 8: LLM Explanation

The system optionally uses a **local Large Language Model through Ollama (Gemma 2B)**.

The LLM generates:

- Explanation of the classification
- Evidence summary

However, the LLM **does not override the rule-based decision logic**, ensuring reliability.

---

## Step 9: Decision Engine

The **Decision Engine** combines outputs from all modules:

- Semantic similarity score
- Intent detection result
- Entity validation result
- Network risk score
- LLM explanation

Based on predefined thresholds, the system assigns the final label:

### 1️⃣ CORRECT
The claim matches verified health information.

### 2️⃣ MISINFORMATION
The claim contradicts verified knowledge or matches known misinformation.

### 3️⃣ UNVERIFIED
There is insufficient evidence to confirm or reject the claim.

---

## Step 10: Output to User

The system displays the result through the web interface.

The output includes:

- Classification label
- Confidence score
- Supporting evidence
- Network risk indicator
- Explanation generated by the system

If misinformation is detected, the system also displays a **warning message**.

---

# Final Workflow Summary

The complete pipeline is:

```
User Input
→ Input Processing
→ Text Preprocessing
→ Knowledge Base Comparison
→ Semantic Similarity Analysis
→ Intent Detection
→ Entity Validation
→ Network Risk Analysis
→ LLM Explanation
→ Decision Engine
→ Final Output
```

This architecture ensures **accurate, explainable, and scalable misinformation detection.**

---

## 🗂️ Project Structure

```
Spread of Misinformation During Public Health Crises COVID-19 : A Network Approach/
│
├── app.py                      # Main Streamlit application
├── classifier.py               # Semantic similarity computation
├── decision_engine.py          # Final classification logic
├── preprocessing.py            # Text cleaning, language detection, translation
├── knowledge_base.py           # Loading verified facts and misinformation patterns
├── intent_detector.py          # Medical claim intent detection
├── entity_validator.py         # Named entity validation using spaCy
├── network_analysis.py         # Claim similarity network (MinHash + LSH)
├── llm_verifier.py             # Local LLM explanation module
├── input_handler.py            # Handles text, URL, and image inputs
├── evaluation.py               # Experimental evaluation framework
├── fact_gatherer.py            # Controlled fact collection
├── knowledge_ingestion.py      # Knowledge base expansion
├── tester_feedback.py          # Tester feedback logging
├── config.py                   # System configuration settings
│
├── data/                       # Knowledge base datasets
│   ├── verified_facts.json
│   ├── known_misinformation.json
│   ├── wikidata_entities.json
│   ├── staged_claims.json
│   └── test_cases.json
│
├── cache/                      # Cached embeddings
│   ├── embeddings_cache.pkl
│   ├── verified_embeddings.pkl
│   ├── misinformation_embeddings.pkl
│   └── wikidata_cache.pkl
│          
└── requirements.txt            # Python dependencies
```

---

## 💻 Tech Stack

## System Components and Technologies

| Component | Technology Used |
|-----------|----------------|
| Web Interface | Streamlit |
| Text Embeddings | Sentence-BERT (all-MiniLM-L6-v2) |
| Local LLM Reasoning | Ollama (Gemma2:2B) |
| Language Detection | langdetect |
| Translation | deep-translator (Google Translate) |
| OCR (Image Text Extraction) | EasyOCR |
| Entity Recognition | spaCy |
| Near-Duplicate Detection | MinHash + LSH (datasketch) |
| Data Storage | JSON Files |

---

## Algorithms

| Category | Algorithm | Module Applied |
|----------|-----------|----------------|
| Semantic Modeling | Sentence-BERT (all-MiniLM-L6-v2) | Semantic Similarity Module |
| Similarity Measurement | Cosine Similarity | Semantic Similarity Module |
| Classification Logic | Threshold-Based Classification | Decision Engine |
| Network Modeling | MinHash (Minimum Hashing) | Network Analysis Module |
| Network Modeling | Locality Sensitive Hashing (LSH) | Network Analysis Module |
| Risk Evaluation | Neighborhood Risk Scoring | Network Analysis Module |
| Image Processing | EasyOCR | Input Handling / Preprocessing |
| Language Processing | Statistical Language Detection (langdetect) | Preprocessing Module |
| Translation | Neural Machine Translation (deep-translator) | Preprocessing Module |
| Text Cleaning | Regex-Based Text Normalization | Preprocessing Module |
| Entity Extraction | spaCy Named Entity Recognition (NER) | Entity Validation Module |
| Intent Classification | Rule-Based Intent Detection | Intent Detection Module |
---

## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Chavva-HasyaReddy/Spread_of_Misinformation_During_Public_Health_Crises_COVID-19_A_Network_Approach.git
cd Spread_of_Misinformation_During_Public_Health_Crises_COVID-19_A_Network_Approach
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Install spaCy language model
```bash
python -m spacy download en_core_web_sm
```

### 4. Install Ollama and load the LLM
Download Ollama:

```
https://ollama.ai/download
```

Pull the required model:

```bash
ollama pull gemma2:2b
```

### 5. Run the application
```bash
streamlit run app.py
```

Visit the application in your browser:

```
http://localhost:8501
```

---

## 🧪 Features Summary

- ✅ Multi-modal claim verification (Text, URL, Image)
- ✅ Semantic similarity based misinformation detection
- ✅ Network-based misinformation pattern detection
- ✅ Intent-aware validation of medical claims
- ✅ Entity validation using structured health data
- ✅ Explainable classification with evidence
- ✅ Offline and privacy-preserving architecture

---

## 🧠 Example Use Cases

- Verify claims like **“Garlic cures COVID-19”**
- Analyze misinformation from **social media posts or articles**
- Upload **screenshots of viral health messages** for verification
- Detect **repeated misinformation patterns** using similarity networks

---

## 🔗 GitHub Repository

Project Source Code: [Spread of Misinformation During Public Health Crises COVID-19 : A Network Approach](https://github.com/Chavva-HasyaReddy/Spread_of_Misinformation_During_Public_Health_Crises_COVID-19_A_Network_Approach)

---

## 📽️ Demo Video

Watch the project demo video here:  
▶️ [Project Demo – Youtube](https://youtu.be/dbLErlGqKI8)

---

## 📄 Project Report

Read the full project report here:  
📘 [Project Report – Google Drive](https://drive.google.com/file/d/1tSb6iqELtSl__Rg7O_xa_VR1z6VrRi-s/view?usp=sharing)

---

## 🤝 Contributors

- [**Chavva Hasya Reddy**](https://github.com/Chavva-HasyaReddy)
- [**Rahman Nayeem Abrar**](https://github.com/Nayeemshaik712)
- [**Gurramkonda Sharun Prakash**](https://github.com/sharunpg)
- [**Chaitanya Bala**](https://github.com/chaitanyabala35)
- [**Jangamreddy Bhavitha Reddy**](https://github.com/J-Bhavitha-Reddy)

---

## 📄 License

This project is developed for **academic and research purposes** as part of a **B.Tech Capstone Project**.
