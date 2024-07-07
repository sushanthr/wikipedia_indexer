import os
import json
import numpy as np
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
from transformers import AutoTokenizer
import faiss
import time
from tqdm import tqdm
from codecs import encode, decode
import re

# Load the ONNX model with DirectML execution
onnx_model_path = "D:\\Projects\\e5-small-v2\\model_opt1_QInt8.onnx"  # Replace with your ONNX model path
options = SessionOptions()
options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
options.intra_op_num_threads = 8
session = InferenceSession(onnx_model_path, options, providers=['DmlExecutionProvider'])

tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small-v2')

def computeEmbeddingsBatch(paragraphs, batch_size=32):
    all_embeddings = []
    
    for i in range(0, len(paragraphs), batch_size):
        batch = paragraphs[i:i+batch_size]
        batch_dict = tokenizer(["passage: " + p for p in batch], max_length=512, padding=True, truncation=True, return_tensors='np')
        
        # Run inference using ONNX Runtime
        ort_inputs = {
            'input_ids': batch_dict['input_ids'].astype(np.int64),
            'attention_mask': batch_dict['attention_mask'].astype(np.int64)
        }
        ort_outputs = session.run(None, ort_inputs)
        
        # Assuming the model output is similar to the PyTorch version
        last_hidden_state = ort_outputs[0]
        embeddings = last_hidden_state.sum(axis=1) / batch_dict['attention_mask'].sum(axis=1)[:, None]
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        all_embeddings.append(embeddings)
    
    return np.concatenate(all_embeddings, axis=0)

def split_text(text, max_length=500):
    chunks = []
    
    # Split on ',', '\n', ';', and '.'
    split_pattern = r'(?<=[,\n;.])\s*'
    sentences = [s.strip() for s in re.split(split_pattern, text) if s.strip()]
    
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += sentence + ' '
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = ""
            
            # If the sentence itself is longer than max_length, split it
            while len(sentence) > max_length:
                split_index = sentence.rfind(' ', 0, max_length)
                if split_index == -1:  # No space found, force split at max_length
                    split_index = max_length
                chunks.append(sentence[:split_index].strip())
                sentence = sentence[split_index:].strip()
            
            current_chunk = sentence + ' '
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Final check to ensure no chunk is longer than max_length
    final_chunks = []
    for chunk in chunks:
        while len(chunk) > max_length:
            split_index = chunk.rfind(' ', 0, max_length)
            if split_index == -1:  # No space found, force split at max_length
                split_index = max_length
            final_chunks.append(chunk[:split_index].strip())
            chunk = chunk[split_index:].strip()
        final_chunks.append(chunk)
    
    return final_chunks

def process_json_files(root_folder, batch_size=32):
    dimension = computeEmbeddingsBatch(["Test sentence"]).shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    file_paths = []
    processing_times = []
    batch_paragraphs = []
    batch_file_paths = []

    total_files = sum([len(files) for _, _, files in os.walk(root_folder) if 'text.json' in files])
    
    with tqdm(total=total_files, desc="Processing files") as pbar:
        for foldername, subfolders, filenames in os.walk(root_folder):
            for filename in filenames:
                if filename == "text.json":
                    file_path = os.path.join(foldername, filename)
                    start_time = time.time()
                    
                    with open(file_path, 'r') as file:
                        file = decode(file.read().strip("\""), 'unicode-escape')
                        data = json.loads(file)
                        text = data['text']
                        chunks = split_text(text)
                        
                        batch_paragraphs.extend(chunks)
                        batch_file_paths.extend([file_path] * len(chunks))
                        
                        if len(batch_paragraphs) >= batch_size:
                            embeddings = computeEmbeddingsBatch(batch_paragraphs, batch_size)
                            index.add(embeddings)
                            file_paths.extend(batch_file_paths)
                            
                            batch_paragraphs = []
                            batch_file_paths = []
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    processing_times.append(processing_time)
                    
                    pbar.update(1)
                    pbar.set_postfix({'Last file time': f'{processing_time:.2f}s'})

    # Process any remaining paragraphs
    if batch_paragraphs:
        embeddings = computeEmbeddingsBatch(batch_paragraphs, batch_size)
        index.add(embeddings)
        file_paths.extend(batch_file_paths)

    return index, file_paths, processing_times

# Usage
root_folder = "archive"
batch_size = 64  # You can adjust this based on your GPU memory
index, file_paths, processing_times = process_json_files(root_folder, batch_size)

# Save the index and file paths
faiss.write_index(index, "embeddings_onnx.index")
with open("file_paths_onnx.json", "w") as f:
    json.dump(file_paths, f)

# Print statistics
print("\nProcessing Statistics:")
print(f"Total files processed: {len(processing_times)}")
print(f"Total processing time: {sum(processing_times):.2f} seconds")
print(f"Average processing time per file: {np.mean(processing_times):.2f} seconds")
print(f"Median processing time: {np.median(processing_times):.2f} seconds")
print(f"Minimum processing time: {min(processing_times):.2f} seconds")
print(f"Maximum processing time: {max(processing_times):.2f} seconds")