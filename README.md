# Google Maps Review Sentiment Analysis Pipeline

**Author:** Zakki Farian  
**Domain:** Natural Language Processing (NLP), Machine Learning  

## Project Overview
This repository contains an end-to-end Machine Learning pipeline designed to classify the sentiment of Indonesian public reviews extracted from Google Maps (specifically targeting the Taman Rimbo Zoo area). The project demonstrates a complete data science workflow, encompassing data acquisition, Large Language Model (LLM) assisted data labeling, text preprocessing, feature extraction, and multi-model evaluation.

## Methodology and Workflow

### 1. Data Acquisition
The raw textual data and associated metadata were extracted from Google Maps. The scraping process was executed utilizing an open-source tool developed by[aliepratama/gmaps-review-scraper](https://github.com/aliepratama/gmaps-review-scraper). 

### 2. LLM-Assisted Sentiment Labeling
Initial exploratory data analysis revealed a significant discrepancy between user-assigned star ratings and the actual textual sentiment (e.g., users assigning a 5-star rating while writing a highly critical review). Consequently, relying on star ratings as the target variable ($y$) would introduce severe label noise.

To establish a mathematically sound ground truth, the dataset was relabeled using a Large Language Model (Google Gemini) integrated directly into a spreadsheet environment. The automated labeling prompt utilized was:
```text
=GEMINI("Analyze the sentiment of this review. Answer with only one word in Indonesian: POSITIF, NEGATIF, or NETRAL. Here is the review: ";[CELL_REFERENCE])
```
This approach ensured that the target labels accurately reflected the semantic polarity of the text.

### 3. Text Preprocessing
Natural language data is inherently unstructured. The pipeline implements a rigorous cleaning process:
* **Noise Reduction:** Regular Expressions (Regex) are applied to strip URLs, mentions, hashtags, numerical digits, and punctuation.
* **Stopword Removal & Stemming:** Utilizing the `Sastrawi` library, high-frequency, low-information words are removed. A custom stopword dictionary was injected to filter out colloquial Indonesian slang. The remaining words are reduced to their morphological roots to decrease the dimensionality of the feature space.

### 4. Feature Engineering (TF-IDF)
The preprocessed corpus is transformed into a sparse numerical matrix using Term Frequency-Inverse Document Frequency (TF-IDF). This statistical measure evaluates the relevance of a word to a document within the entire corpus.
$$ \text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \left( \log \frac{N}{1 + \text{df}(t)} + 1 \right) $$

### 5. Algorithmic Modeling and Evaluation
To combat class imbalance within the dataset, the pipeline trains and evaluates three distinct algorithms, all configured with balanced class weights:
1. **Stochastic Gradient Descent (SGD) Classifier:** Optimizes the logistic loss function iteratively.
2. **Support Vector Machine (SVM):** Constructs an optimal hyperplane that maximizes the margin between classes.
3. **Random Forest:** An ensemble learning method utilizing multiple decision trees to reduce variance.

The models are evaluated using a Stratified 10-Fold Cross-Validation approach and an isolated testing set. Performance is quantified using Accuracy, Precision, Recall, and the F1-Score.

## Repository Structure
* `notebooks/` : Contains the primary Jupyter Notebook detailing the OOP-based pipeline architecture.
* `data/` : Contains the raw and preprocessed CSV datasets.
* `README.md` : Project documentation.

## Technologies Used
* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Natural Language Processing:** Sastrawi, NLTK
* **Visualization:** Matplotlib, Seaborn, WordCloud

## Acknowledgments
* Data scraping module provided by [aliepratama](https://github.com/aliepratama/gmaps-review-scraper).
