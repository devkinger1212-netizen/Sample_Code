# End-to-End NLP & Predictive Modeling Pipeline

### **Project Overview**
This repository contains a full Python data pipeline engineered to process unstructured text, perform thematic clustering, and execute sentence-weighted sentiment analysis. Originally developed to extract predictive signals and classify textual data, this codebase demonstrates an end-to-end workflow from raw data preprocessing to the deployment of machine learning models and interactive visual reporting.

### **Relevance to Empirical Research**
While this specific implementation utilizes Natural Language Processing (NLP) and predictive machine learning, the underlying architecture demonstrates the core technical competencies required for complex empirical economics research:
* **Advanced Data Wrangling:** The ability to clean, merge, and structure messy, high-dimensional datasets.
* **Algorithmic Problem Solving:** Translating abstract concepts (e.g., text semantics) into quantifiable, analyzable metrics.
* **Reproducible Workflows:** Writing modular, well-documented, and efficient code that cleanly separates data processing, modeling, and visualization.

### **Pipeline Architecture**
The codebase is modularized into three core analytical tracks to ensure reproducibility and clean execution:

#### **1. `Code_Sentiment_FinBERT.py` (Text Processing & Sentiment Engine)**
* Cleans raw text inputs, stripping HTML artifacts and applying regular expressions for robust sentence tokenization.
* Implements a pre-trained transformer model (`ProsusAI/finbert`) via the Hugging Face pipeline.
* Calculates a nuanced, sentence-length-weighted net sentiment score for entire textual sections to prevent short, highly polarized sentences from skewing aggregated results.

#### **2. `Code_Cluster_Sentiment.py` (Thematic Clustering Pipeline)**
* Utilizes `spaCy` to extract meaningful lemmas (nouns and adjectives) while filtering out boilerplate and stopwords.
* Embeds extracted vocabulary into mathematical vectors using `SentenceTransformers` (`all-MiniLM-L6-v2`).
* Applies K-Means clustering to group semantically similar terminology into distinct thematic clusters.
* Maps these thematic clusters back to the original text corpus and scores the contextual sentences, allowing for highly granular, entity-x-year-x-cluster sentiment tracking.

#### **3. `Code_Charts_Visualization.py` (Data Aggregation & Visual Translation)**
* Processes and aggregates the high-dimensional outputs from the modeling scripts using `pandas` and `numpy`.
* Utilizes `matplotlib` and `seaborn` to generate stakeholder-ready visualizations, including:
  * Waterfall charts mapping sentiment deterioration.
  * Heatmaps displaying cross-sectional risk intensity.
  * Granular cluster-landscape visualizations mapping structural shifts over time.

### **Technical Stack**
* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning / Clustering:** `scikit-learn` (K-Means)
* **Natural Language Processing:** `transformers` (Hugging Face), `spacy`, `sentence-transformers`, `nltk`, `re`
* **Visualization:** `matplotlib`, `seaborn`

### **A Note on Data and Execution**
To respect corporate confidentiality agreements, all proprietary text and company-specific datasets have been removed. The code provided here serves as a functional demonstration of the underlying analytical architecture and pipeline logic. To execute the scripts, users must supply an appropriately structured input dataset (e.g., standard `.xlsx` files containing raw text columns) and configure the local file paths defined at the top of each script.
