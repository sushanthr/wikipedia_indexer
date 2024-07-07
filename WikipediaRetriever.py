import faiss
import numpy as np
import json
import os
from onnxruntime import InferenceSession
from transformers import AutoTokenizer, AutoModel
from codecs import encode, decode
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small-v2')
model = AutoModel.from_pretrained('intfloat/e5-small-v2')

def computeEmbeddings(paragraph):
    batch_dict = tokenizer("query: " + paragraph, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**batch_dict)
    embeddings = outputs.last_hidden_state.sum(dim=1) / batch_dict['attention_mask'].sum(dim=1)[..., None]
    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings[0]

def load_index_and_paths():
    index = faiss.read_index("embeddings.index")
    with open("file_paths.json", "r") as f:
        file_paths = json.load(f)
    return index, file_paths

def get_json_content(file_path):
    with open(file_path, 'r') as f:
        file = decode(f.read().strip("\""), 'unicode-escape')
        return json.loads(file)

def search_similar_documents(question, index, file_paths, k=5):
    question_embedding = computeEmbeddings(question)
    _, indices = index.search(np.array([question_embedding.detach().numpy()]), k)
    
    results = []
    for idx in indices[0]:
        file_path = file_paths[idx]
        json_content = get_json_content(file_path)
        results.append({
            'title': json_content.get('title', 'No title'),
            'url': json_content.get('url', 'No URL'),
            'text': json_content.get('text', 'No text')[:500]  # First 500 characters of text
        })
    
    return results

def main():
    index, file_paths = load_index_and_paths()
    
    while True:
        question = input("Enter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        
        results = search_similar_documents(question, index, file_paths)
        
        print("\nTop 5 relevant documents:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Title: {result['title']}")
            print(f"   URL: {result['url']}")
            print(f"   Text snippet: {result['text'][:200]}...")  # First 200 characters

if __name__ == "__main__":
    main()