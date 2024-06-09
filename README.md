### News Article Processor Documentation

#### Introduction

This documentation provides a detailed explanation of the `NewsArticleProcessor` class, which processes news articles from a JSON file, computes embeddings, and retrieves relevant articles based on a query. The goal is to create a clean, modular, and efficient codebase for handling and querying news articles using embeddings.

### Code with Thought Process

#### Step 1: Initialization


- Set up the class to handle the core functionalities, such as loading JSON data, processing articles, computing embeddings, and saving/loading data.
- Use fixed values for the embedding model (`all-mpnet-base-v2`) and the CSV save path to simplify the usage.

```python
import os
import re
import json
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util
from time import perf_counter as timer

class NewsArticleProcessor:
    def __init__(self, json_path, min_token_length=30, n_resources_to_return=10):
        self.json_path = json_path
        self.embedding_model_name = "all-mpnet-base-v2"
        self.min_token_length = min_token_length
        self.n_resources_to_return = n_resources_to_return
        self.embedding_model = SentenceTransformer(model_name_or_path=self.embedding_model_name, device="cpu")
        self.df = None
        self.articles_and_chunks = []
        self.save_path = os.path.join(os.getcwd(), 'articles_and_embeddings_df.csv')
```

#### Step 2: Loading JSON Data


- Load the JSON data from the specified file path. This is the initial step to get the raw data.

```python
    def load_json(self):
        with open(self.json_path, 'r') as json_file:
            data = json.load(json_file)
        return data
```

#### Step 3: Cleaning Text Data


- Clean the text data by removing newline characters. This helps in ensuring that the text data is in a consistent format for further processing.

```python
    @staticmethod
    def clean_text(text: str) -> str:
        return text.replace('\n', '')
```

#### Step 4: Preprocessing Data


- Combine the title and body of each article to create a single text field for embedding.
- Create a DataFrame for easier manipulation and computation.

```python
    def preprocess_data(self, data):
        for article in data:
            article['articleBody'] = self.clean_text(article['articleBody'])
        self.df = pd.DataFrame(data)
        self.df['joined_article'] = self.df['title'] + ' ' + self.df['articleBody']
```

#### Step 5: Processing Articles


- Process each article to compute character count, word count, and token count.
- Store these counts for filtering and further analysis.

```python
    def process_articles(self):
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            article = row['joined_article']
            chunk_dict = {}

            article = article.replace("  ", " ").strip()    
            article = re.sub(r'\.([A-Z])', r'. \1', article)
            
            chunk_dict["joined_article"] = article
            chunk_dict["article_char_count"] = len(article)
            chunk_dict["article_word_count"] = len(article.split())
            chunk_dict["article_token_count"] = len(article) / 4
            self.articles_and_chunks.append(chunk_dict)
        
        self.df = pd.DataFrame(self.articles_and_chunks)
```

#### Step 6: Filtering and Embedding Articles


- Filter articles by token count to remove very short articles.
- Compute embeddings for the filtered articles using the SentenceTransformer model.

```python
    def filter_and_embed_articles(self):
        articles_over_min_token_len = self.df[self.df["article_token_count"] > self.min_token_length].to_dict(orient="records")

        for item in tqdm(articles_over_min_token_len):
            item["embedding"] = self.embedding_model.encode(item["joined_article"])

        self.df = pd.DataFrame(articles_over_min_token_len)
```

#### Step 7: Saving and Loading Processed Data


- Save the processed data with embeddings to a CSV file.
- Provide functionality to load the processed data from the CSV file.

```python
    def save_to_csv(self):
        self.df.to_csv(self.save_path, index=False)

    def load_from_csv(self):
        self.df = pd.read_csv(self.save_path)
        return self.df
```

#### Step 8: Preparing Embeddings for Querying


- Convert the embedding strings back to numpy arrays.
- Create a tensor of embeddings for efficient querying.

```python
    def prepare_embeddings(self):
        self.df["embedding"] = self.df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        self.articles = self.df.to_dict(orient="records")
        self.embeddings = torch.tensor(np.array(self.df["embedding"].tolist()), dtype=torch.float32).to("cpu")
```

#### Step 9: Retrieving Relevant Articles


- Take a query string and compute its embedding.
- Use dot product scores to find the most relevant articles based on the query embedding.

```python
    def retrieve_relevant_resources(self, query: str, print_time: bool=True):
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)

        start_time = timer()
        dot_scores = util.dot_score(query_embedding, self.embeddings)[0]
        end_time = timer()

        if print_time:
            print(f"[INFO] Time taken to get scores on {len(self.embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

        scores, indices = torch.topk(input=dot_scores, k=self.n_resources_to_return)
        relevant_articles = [self.articles[index]["joined_article"] for index in indices]

        return relevant_articles
```

### QuestionAnswer Class Documentation

#### Introduction

The `QuestionAnswer` class is responsible for generating answers to questions based on a given context. It uses a pre-trained T5 model for conditional generation.

### Code Overview

The `QuestionAnswer` class encapsulates functionalities related to question answering:

1. **Initialization**: 
   - The class is initialized with the model name of the T5 model.

2. **Answer Generation**:
   - The `generate_answer()` method takes a question and context as input, encodes them using the T5 tokenizer, generates an answer using the T5 model, and decodes the answer tokens into human-readable text.

### Usage

Both the `NewsArticleProcessor` and `QuestionAnswer` classes can be used together to process news articles and generate answers to questions based on the processed articles.

```python
processor = NewsArticleProcessor(json_path='news.json')
processor.process(load_csv=True)
processor.prepare_embeddings()
relevant_articles = processor.retrieve_relevant_resources(query='Your query here')

qa = QuestionAnswer(model_name="t5-base")
context_text = processor.array_to_string_with_newlines(relevant_articles)
answer = qa.generate_answer(question='Your question here', context=context_text)
print("Answer:", answer)
```

### Dependencies

Ensure that the following dependencies are installed:

- pandas
- tqdm
- sentence-transformers
- torch
- numpy

These can be installed via pip using:

```
pip install -r requirements.txt
```
