#!/usr/bin/env python3
"""
Prepare personal text corpus for language model training
"""
import os
import re
import argparse
from pathlib import Path

def clean_text(text):
    """Clean and normalize text for language model training"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\'\"]', ' ', text)
    
    # Split into sentences (roughly)
    sentences = re.split(r'[\.!?]+', text)
    
    # Clean each sentence
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 3:  # Only keep sentences with some content
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences

def process_file(file_path):
    """Process a single text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return clean_text(content)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Prepare personal corpus for LM training')
    parser.add_argument('--input_dir', required=True, help='Directory containing your text files')
    parser.add_argument('--output_file', required=True, help='Output corpus file')
    parser.add_argument('--file_extensions', default='txt,md,py,js,html,csv', 
                       help='Comma-separated file extensions to process')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    extensions = args.file_extensions.split(',')
    
    all_sentences = []
    
    # Process all files with specified extensions
    for ext in extensions:
        for file_path in input_dir.rglob(f'*.{ext}'):
            print(f"Processing: {file_path}")
            sentences = process_file(file_path)
            all_sentences.extend(sentences)
    
    # Write processed corpus
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for sentence in all_sentences:
            f.write(sentence + '\n')
    
    print(f"Processed {len(all_sentences)} sentences")
    print(f"Corpus saved to: {args.output_file}")

if __name__ == "__main__":
    main()
