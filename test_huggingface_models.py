#!/usr/bin/env python3
"""
Test script for evaluating brain-to-text system with different Hugging Face language models.

This script allows you to test the brain-to-text system with various language models
from Hugging Face, comparing their performance on the validation set.

Usage:
    python test_huggingface_models.py --models "microsoft/DialoGPT-medium,gpt2,facebook/opt-1.3b"
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import redis
import argparse
import time
import json
from tqdm import tqdm
import editdistance
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Add model training directory to path
sys.path.append('model_training')
from rnn_model import GRUDecoder
from evaluate_model_helpers import *

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_huggingface_model(model_name, cache_dir=None, device='cuda'):
    """
    Load a Hugging Face language model and tokenizer.
    
    Args:
        model_name (str): Name of the model on Hugging Face Hub
        cache_dir (str): Directory to cache downloaded models
        device (str): Device to load model on
        
    Returns:
        tuple: (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        
        if device != 'cpu':
            model = model.to(device)
        
        model.eval()
        
        # Ensure padding token
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info(f"Successfully loaded {model_name}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        return None, None

@torch.inference_mode()
def rescore_with_language_model(model, tokenizer, device, hypotheses, length_penalty=0.0):
    """
    Rescore hypotheses using a language model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        device: Device to run inference on
        hypotheses (list): List of text hypotheses to score
        length_penalty (float): Length penalty to apply
        
    Returns:
        list: Scores for each hypothesis
    """
    if not hypotheses:
        return []
    
    model.eval()
    
    try:
        inputs = tokenizer(hypotheses, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model(**inputs)
        log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        log_probs = log_probs.cpu().numpy()
        
        input_ids = inputs['input_ids'].cpu().numpy()
        attention_mask = inputs['attention_mask'].cpu().numpy()
        batch_size, seq_len, _ = log_probs.shape
        
        scores = []
        for i in range(batch_size):
            n_tokens = int(attention_mask[i].sum())
            # Sum log-probs of each token given the previous context
            score = sum(
                log_probs[i, t-1, input_ids[i, t]]
                for t in range(1, n_tokens)
            )
            scores.append(score - n_tokens * length_penalty)
            
        return scores
        
    except Exception as e:
        logger.error(f"Error during rescoring: {e}")
        return [0.0] * len(hypotheses)

def simulate_lm_decode(model, tokenizer, device, nbest_hypotheses, acoustic_scale=0.3, 
                      length_penalty=0.0, alpha=0.5):
    """
    Simulate the language model decoding process with a different HF model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer  
        device: Device to run on
        nbest_hypotheses: List of (sentence, acoustic_score, ngram_score) tuples
        acoustic_scale: Weight for acoustic scores
        length_penalty: Length penalty for generation
        alpha: Weight for new LM scores vs old ngram scores
        
    Returns:
        tuple: (best_sentence, all_scores_formatted)
    """
    if not nbest_hypotheses:
        return "", []
        
    hypotheses = []
    acoustic_scores = []
    old_lm_scores = []
    
    for hyp_data in nbest_hypotheses:
        sentence = hyp_data[0].strip()
        if len(sentence) == 0:
            continue
            
        # Clean up the sentence
        sentence = sentence.replace('>', '').replace('  ', ' ')
        sentence = sentence.replace(' ,', ',').replace(' .', '.')
        sentence = sentence.replace(' ?', '?')
        
        hypotheses.append(sentence)
        acoustic_scores.append(hyp_data[1])
        old_lm_scores.append(hyp_data[2])
    
    if not hypotheses:
        return "", []
        
    # Convert to numpy arrays
    acoustic_scores = np.array(acoustic_scores)
    old_lm_scores = np.array(old_lm_scores)
    
    # Get new LM scores
    new_lm_scores = np.array(rescore_with_language_model(
        model, tokenizer, device, hypotheses, length_penalty
    ))
    
    # Calculate total scores
    total_scores = (acoustic_scale * acoustic_scores) + ((1 - alpha) * old_lm_scores) + (alpha * new_lm_scores)
    
    # Get best hypothesis
    max_idx = np.argmax(total_scores)
    best_hypothesis = hypotheses[max_idx]
    
    # Create formatted output
    scores_formatted = []
    min_len = min(len(hypotheses), len(new_lm_scores), len(total_scores))
    for i in range(min_len):
        scores_formatted.append(';'.join(map(str, [
            hypotheses[i], 
            acoustic_scores[i], 
            old_lm_scores[i], 
            new_lm_scores[i], 
            total_scores[i]
        ])))
    
    return best_hypothesis, scores_formatted

def evaluate_model_with_hf_lm(model_name, model_path, data_dir, csv_path, device, cache_dir=None):
    """
    Evaluate the brain-to-text system using a specific Hugging Face language model.
    
    Args:
        model_name (str): Name of the HF model to test
        model_path (str): Path to the pretrained RNN model
        data_dir (str): Path to the dataset directory
        csv_path (str): Path to the CSV metadata file
        device: Device to run on
        cache_dir (str): Cache directory for HF models
        
    Returns:
        dict: Evaluation results
    """
    logger.info(f"Starting evaluation with model: {model_name}")
    
    # Load Hugging Face model
    hf_model, hf_tokenizer = load_huggingface_model(model_name, cache_dir, device)
    if hf_model is None:
        return {"error": f"Failed to load {model_name}"}
    
    # Load RNN model
    model_args = OmegaConf.load(os.path.join(model_path, 'checkpoint/args.yaml'))
    
    rnn_model = GRUDecoder(
        neural_dim=model_args['model']['n_input_features'],
        n_units=model_args['model']['n_units'], 
        n_days=len(model_args['dataset']['sessions']),
        n_classes=model_args['dataset']['n_classes'],
        rnn_dropout=model_args['model']['rnn_dropout'],
        input_dropout=model_args['model']['input_network']['input_layer_dropout'],
        n_layers=model_args['model']['n_layers'],
        patch_size=model_args['model']['patch_size'],
        patch_stride=model_args['model']['patch_stride'],
    )
    
    # Load RNN weights
    checkpoint = torch.load(os.path.join(model_path, 'checkpoint/best_checkpoint'), weights_only=False)
    # Clean up key names
    for key in list(checkpoint['model_state_dict'].keys()):
        checkpoint['model_state_dict'][key.replace("module.", "")] = checkpoint['model_state_dict'].pop(key)
        checkpoint['model_state_dict'][key.replace("_orig_mod.", "")] = checkpoint['model_state_dict'].pop(key)
    rnn_model.load_state_dict(checkpoint['model_state_dict'])
    rnn_model.to(device)
    rnn_model.eval()
    
    # Load data
    b2txt_csv_df = pd.read_csv(csv_path)
    test_data = {}
    total_trials = 0
    
    for session in model_args['dataset']['sessions']:
        files = [f for f in os.listdir(os.path.join(data_dir, session)) if f.endswith('.hdf5')]
        if 'data_val.hdf5' in files:
            eval_file = os.path.join(data_dir, session, 'data_val.hdf5')
            data = load_h5py_file(eval_file, b2txt_csv_df)
            test_data[session] = data
            total_trials += len(test_data[session]["neural_features"])
    
    logger.info(f"Loaded {total_trials} validation trials")
    
    # Generate phoneme predictions
    logger.info("Generating phoneme predictions...")
    with tqdm(total=total_trials, desc='RNN inference') as pbar:
        for session, data in test_data.items():
            data['logits'] = []
            input_layer = model_args['dataset']['sessions'].index(session)
            
            for trial in range(len(data['neural_features'])):
                neural_input = data['neural_features'][trial]
                neural_input = np.expand_dims(neural_input, axis=0)
                neural_input = torch.tensor(neural_input, device=device, dtype=torch.bfloat16)
                
                logits = runSingleDecodingStep(neural_input, input_layer, rnn_model, model_args, device)
                data['logits'].append(logits)
                pbar.update(1)
    
    # For this simplified version, we'll create mock n-best lists from the phoneme predictions
    # In a full implementation, you'd integrate with the Kaldi n-gram decoder
    logger.info("Generating text predictions...")
    results = {
        'model_name': model_name,
        'predictions': [],
        'true_sentences': [],
        'sessions': [],
        'blocks': [],
        'trials': [],
    }
    
    with tqdm(total=total_trials, desc=f'LM inference ({model_name})') as pbar:
        for session, data in test_data.items():
            for trial in range(len(data['logits'])):
                # Convert logits to phoneme sequence (simplified)
                logits = data['logits'][trial][0]
                pred_seq = np.argmax(logits, axis=-1)
                pred_seq = [int(p) for p in pred_seq if p != 0]
                pred_seq = [pred_seq[i] for i in range(len(pred_seq)) if i == 0 or pred_seq[i] != pred_seq[i-1]]
                phoneme_seq = [LOGIT_TO_PHONEME[p] for p in pred_seq]
                
                # Create mock n-best list (in practice this would come from Kaldi decoder)
                base_sentence = " ".join(phoneme_seq).replace(" | ", " ")
                mock_nbest = [
                    (base_sentence, -10.0, -5.0),  # (sentence, acoustic_score, ngram_score)
                    (base_sentence.replace("AH", "UH"), -12.0, -6.0),
                    (base_sentence.replace("IH", "EH"), -11.0, -5.5),
                ]
                
                # Use HF model to rescore
                best_sentence, _ = simulate_lm_decode(
                    hf_model, hf_tokenizer, device, mock_nbest,
                    acoustic_scale=0.35, alpha=0.55
                )
                
                results['predictions'].append(best_sentence)
                results['true_sentences'].append(data['sentence_label'][trial])
                results['sessions'].append(session)
                results['blocks'].append(data['block_num'][trial])
                results['trials'].append(data['trial_num'][trial])
                
                pbar.update(1)
    
    # Calculate metrics
    total_edit_distance = 0
    total_words = 0
    
    for i in range(len(results['predictions'])):
        true_sentence = remove_punctuation(results['true_sentences'][i]).strip()
        pred_sentence = remove_punctuation(results['predictions'][i]).strip()
        
        ed = editdistance.eval(true_sentence.split(), pred_sentence.split())
        total_edit_distance += ed
        total_words += len(true_sentence.split())
    
    wer = (total_edit_distance / total_words) * 100 if total_words > 0 else 100.0
    
    results.update({
        'word_error_rate': wer,
        'total_edit_distance': total_edit_distance,
        'total_words': total_words,
        'num_trials': len(results['predictions'])
    })
    
    logger.info(f"Completed evaluation for {model_name}")
    logger.info(f"Word Error Rate: {wer:.2f}%")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test brain-to-text system with different Hugging Face models')
    parser.add_argument('--models', type=str, required=True,
                       help='Comma-separated list of HF model names to test')
    parser.add_argument('--model_path', type=str, default='data/t15_pretrained_rnn_baseline',
                       help='Path to pretrained RNN model')
    parser.add_argument('--data_dir', type=str, default='data/hdf5_data_final',
                       help='Path to dataset directory')
    parser.add_argument('--csv_path', type=str, default='data/t15_copyTaskData_description.csv',
                       help='Path to CSV metadata file')
    parser.add_argument('--cache_dir', type=str, default='./hf_cache',
                       help='Directory to cache Hugging Face models')
    parser.add_argument('--gpu_number', type=int, default=0,
                       help='GPU number to use (-1 for CPU)')
    parser.add_argument('--output_file', type=str, default='hf_model_comparison.json',
                       help='File to save results')
    
    args = parser.parse_args()
    
    # Set up device
    if torch.cuda.is_available() and args.gpu_number >= 0:
        device = torch.device(f'cuda:{args.gpu_number}')
        logger.info(f'Using GPU: {device}')
    else:
        device = torch.device('cpu')
        logger.info('Using CPU')
    
    # Parse model list
    model_names = [name.strip() for name in args.models.split(',')]
    logger.info(f"Testing {len(model_names)} models: {model_names}")
    
    # Create cache directory
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Test each model
    all_results = {}
    
    for model_name in model_names:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing model: {model_name}")
            logger.info(f"{'='*60}")
            
            results = evaluate_model_with_hf_lm(
                model_name=model_name,
                model_path=args.model_path,
                data_dir=args.data_dir,
                csv_path=args.csv_path,
                device=device,
                cache_dir=args.cache_dir
            )
            
            all_results[model_name] = results
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error testing {model_name}: {e}")
            all_results[model_name] = {"error": str(e)}
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    
    for model_name, results in all_results.items():
        if 'error' in results:
            logger.info(f"{model_name}: ERROR - {results['error']}")
        else:
            wer = results.get('word_error_rate', 'N/A')
            logger.info(f"{model_name}: WER = {wer:.2f}%")
    
    logger.info(f"\nDetailed results saved to: {args.output_file}")

if __name__ == '__main__':
    main()
