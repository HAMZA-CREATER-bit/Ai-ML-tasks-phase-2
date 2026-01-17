# Ai-ML-tasks-phase-2


Task 1: News Topic Classification with BERT

Objective:
Fine-tune a pre-trained BERT transformer to classify news headlines into categories: World, Sports, Business, Sci/Tech.

Key Steps:

* Load and preprocess the AG News dataset
* Tokenize headlines using BERT tokenizer
* Fine-tune `bert-base-uncased` model
* Evaluate model using Accuracy and F1-score
* Quick predictions for sample headlines

Technologies: Python, PyTorch, HuggingFace Transformers, Datasets

---

## **Task 2: Customer Churn Prediction Pipeline**

**Objective:**
Build a production-ready machine learning pipeline to predict customer churn.

**Key Steps:**

* Load and preprocess Telco Churn dataset (numerical & categorical features)
* Construct scikit-learn pipeline with scaling, encoding, and classifier
* Train models: Logistic Regression & Random Forest
* Hyperparameter tuning using GridSearchCV
* Evaluate using Accuracy, F1-score, Confusion Matrix
* Save pipeline with `joblib` for reuse

**Technologies:** Python, scikit-learn, Pandas, Joblib

---

## **Task 3: PDF-based Q&A with LLM & FAISS**

**Objective:**
Develop an interactive system to answer questions from PDF documents using embeddings and retrieval.

**Key Steps:**

* Load and split PDF documents into chunks
* Generate embeddings using HuggingFace models
* Create FAISS vector database for retrieval
* Build RetrievalQA chain using LangChain
* Deploy interactive Streamlit interface for Q&A

**Technologies:** Python, LangChain, HuggingFace Embeddings, FAISS, Streamlit

---

## **How to Run**

1. Clone this repository:

```bash
git clone <repo-link>
cd <repo-folder>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. **Task 1 (BERT News Classifier)**

```bash
# Run the notebook or Python script
python task1_news_classifier.py
```

4. **Task 2 (Churn Pipeline)**

```bash
python task2_churn_pipeline.py
```

5. **Task 3 (PDF Q&A)**

```bash
streamlit run task3_pdf_qa.py
```

---

## **Results & Outputs**

* Task 1: Model fine-tuned, Accuracy & F1-score displayed
* Task 2:Best ML pipeline saved (`.pkl` file) and metrics displayed
* Task 3: Interactive PDF Q&A system with live answers via Streamlit

---

## Key Skills Gained

* NLP with Transformers (BERT)
* ML pipeline design and hyperparameter tuning
* Document embeddings and retrieval-based LLM
* Model evaluation & deployment
* Interactive AI application development

---

## Technologies Used**

Python | PyTorch | HuggingFace Transformers | scikit-learn | Pandas | LangChain | FAISS | Streamlit


