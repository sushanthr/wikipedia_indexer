import re

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
