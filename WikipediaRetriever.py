import faiss
import numpy as np
import json
import os
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
from transformers import AutoTokenizer
from codecs import encode, decode
from Utils import *

# Load the ONNX model with DirectML execution
onnx_model_path = "D:\\Projects\\e5-small-v2\\model_opt1_QInt8.onnx"  # Replace with your ONNX model path
options = SessionOptions()
options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
options.intra_op_num_threads = 8
session = InferenceSession(onnx_model_path, options, providers=['DmlExecutionProvider'])

tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small-v2')

def computeEmbeddings(paragraph):
    batch_dict = tokenizer("query: " + paragraph, max_length=512, padding=True, truncation=True, return_tensors='np')
    
    ort_inputs = {
        'input_ids': batch_dict['input_ids'].astype(np.int64),
        'attention_mask': batch_dict['attention_mask'].astype(np.int64)
    }
    ort_outputs = session.run(None, ort_inputs)
    
    return ort_outputs[1]

def load_index_and_paths():
    index = faiss.read_index("embeddings_onnx.index")
    with open("file_paths_and_indices_onnx.json", "r") as f:
        data = json.load(f)
    return index, data['file_paths'], data['paragraph_indices']

def get_json_content(file_path):
    with open(file_path, 'r') as f:
        file = decode(f.read().strip("\""), 'unicode-escape')
        return json.loads(file)

def search_similar_documents(question, index, file_paths, paragraph_indices, k=5):
    question_embedding = computeEmbeddings(question)
    _, indices = index.search(np.array(question_embedding), k)
    
    results = []
    for idx in indices[0]:
        file_path = file_paths[idx]
        paragraph_index = paragraph_indices[idx]
        json_content = get_json_content(file_path)
        
        # Split the text into paragraphs
        paragraphs = split_text(json_content.get('text', ''));
        
        # Get the relevant paragraph
        relevant_paragraph = paragraphs[paragraph_index] if paragraph_index < len(paragraphs) else "Paragraph not found"
        
        results.append({
            'title': json_content.get('title', 'No title'),
            'url': json_content.get('url', 'No URL'),
            'text_snippet': relevant_paragraph
        })
    
    return results

def main():
    index, file_paths, paragraph_indices = load_index_and_paths()
    
    while True:
        question = input("Enter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        
        results = search_similar_documents(question, index, file_paths, paragraph_indices)
        
        print("\nTop 5 relevant documents:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Title: {result['title']}")
            print(f"   URL: {result['url']}")
            print(f"   Text snippet: {result['text_snippet'][:200]}...")  # First 200 characters of the relevant paragraph

if __name__ == "__main__":
    main()