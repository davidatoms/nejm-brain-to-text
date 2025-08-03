#!/usr/bin/env python3
"""
Advanced script for testing brain-to-text system with different Hugging Face models.

This script integrates with the existing Kaldi n-gram language model pipeline,
replacing only the OPT rescoring component with different Hugging Face models.

Usage:
    # First start the language model server in another terminal:
    cd language_model
    conda activate b2txt25_lm
    python language-model-standalone.py --lm_path pretrained_language_models/openwebtext_1gram_lm_sil --do_opt

    # Then run this script:
    python test_hf_with_kaldi_integration.py --models "gpt2,microsoft/DialoGPT-medium,facebook/opt-1.3b"
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
import subprocess
import signal
from contextlib import contextmanager

# Add model training directory to path
sys.path.append('model_training')
from rnn_model import GRUDecoder
from evaluate_model_helpers import *

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HuggingFaceLanguageModelServer:
    """
    Custom language model server that replaces OPT in the existing pipeline.
    """
    
    def __init__(self, model_name, cache_dir=None, device='cuda'):
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading Hugging Face model: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            
            if device != 'cpu':
                self.model = self.model.to(device)
            
            self.model.eval()
            
            # Ensure padding token
            self.tokenizer.padding_side = "right"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"Successfully loaded {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise
    
    @torch.inference_mode()
    def rescore_hypotheses(self, hypotheses, length_penalty=0.0):
        """
        Rescore a list of text hypotheses using the language model.
        
        Args:
            hypotheses (list): List of text strings to score
            length_penalty (float): Length penalty to apply
            
        Returns:
            list: Log probability scores for each hypothesis
        """
        if not hypotheses:
            return []
        
        try:
            inputs = self.tokenizer(hypotheses, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
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

def create_custom_language_model_script(model_name, cache_dir, gpu_number, temp_script_path):
    """
    Create a custom version of the language model script that uses a different HF model.
    """
    
    # Read the original script
    original_script_path = 'language_model/language-model-standalone.py'
    with open(original_script_path, 'r') as f:
        script_content = f.read()
    
    # Replace the OPT model loading with our custom model
    model_loading_replacement = f'''
def build_opt(
        model_name='{model_name}',
        cache_dir='{cache_dir}',
        device='cuda' if torch.cuda.is_available() else 'cpu',
    ):
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
    )

    if device != 'cpu':
        # Move the model to the GPU
        model = model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # ensure padding token
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
'''
    
    # Replace the build_opt function
    import re
    pattern = r'def build_opt\(.*?\n    return model, tokenizer'
    script_content = re.sub(pattern, model_loading_replacement.strip(), script_content, flags=re.DOTALL)
    
    # Write the modified script
    with open(temp_script_path, 'w') as f:
        f.write(script_content)
    
    logger.info(f"Created custom language model script: {temp_script_path}")

@contextmanager
def language_model_server(model_name, cache_dir, gpu_number=0):
    """
    Context manager to start and stop a language model server process.
    """
    temp_script_path = f'temp_lm_script_{model_name.replace("/", "_")}.py'
    
    try:
        # Create custom script
        create_custom_language_model_script(model_name, cache_dir, gpu_number, temp_script_path)
        
        # Start the language model server
        cmd = [
            'python', temp_script_path,
            '--lm_path', 'language_model/pretrained_language_models/openwebtext_1gram_lm_sil',
            '--do_opt',
            '--gpu_number', str(gpu_number),
            '--cache_dir', cache_dir
        ]
        
        logger.info(f"Starting language model server with {model_name}...")
        process = subprocess.Popen(cmd, cwd='.')
        
        # Wait a bit for the server to start
        time.sleep(10)
        
        # Check if redis is accessible
        r = redis.Redis(host='localhost', port=6379, db=0)
        for _ in range(30):  # Wait up to 30 seconds
            try:
                r.ping()
                logger.info("Language model server is ready")
                break
            except redis.exceptions.ConnectionError:
                time.sleep(1)
        else:
            raise RuntimeError("Could not connect to language model server")
        
        yield process
        
    finally:
        # Clean up
        if 'process' in locals():
            logger.info("Stopping language model server...")
            process.terminate()
            time.sleep(2)
            if process.poll() is None:
                process.kill()
            process.wait()
        
        # Remove temporary script
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)

def evaluate_with_kaldi_integration(model_name, model_path, data_dir, csv_path, device, cache_dir, gpu_number=0):
    """
    Evaluate the brain-to-text system using a specific Hugging Face model integrated with Kaldi.
    """
    logger.info(f"Starting evaluation with model: {model_name}")
    
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
    for key in list(checkpoint['model_state_dict'].keys()):\n        checkpoint['model_state_dict'][key.replace("module.", "")] = checkpoint['model_state_dict'].pop(key)\n        checkpoint['model_state_dict'][key.replace("_orig_mod.", "")] = checkpoint['model_state_dict'].pop(key)\n    rnn_model.load_state_dict(checkpoint['model_state_dict'])\n    rnn_model.to(device)\n    rnn_model.eval()\n    \n    # Load data\n    b2txt_csv_df = pd.read_csv(csv_path)\n    test_data = {}\n    total_trials = 0\n    \n    for session in model_args['dataset']['sessions']:\n        files = [f for f in os.listdir(os.path.join(data_dir, session)) if f.endswith('.hdf5')]\n        if 'data_val.hdf5' in files:\n            eval_file = os.path.join(data_dir, session, 'data_val.hdf5')\n            data = load_h5py_file(eval_file, b2txt_csv_df)\n            test_data[session] = data\n            total_trials += len(test_data[session]["neural_features"])\n    \n    logger.info(f"Loaded {total_trials} validation trials")\n    \n    # Generate phoneme predictions\n    logger.info("Generating phoneme predictions...")\n    with tqdm(total=total_trials, desc='RNN inference') as pbar:\n        for session, data in test_data.items():\n            data['logits'] = []\n            input_layer = model_args['dataset']['sessions'].index(session)\n            \n            for trial in range(len(data['neural_features'])):\n                neural_input = data['neural_features'][trial]\n                neural_input = np.expand_dims(neural_input, axis=0)\n                neural_input = torch.tensor(neural_input, device=device, dtype=torch.bfloat16)\n                \n                logits = runSingleDecodingStep(neural_input, input_layer, rnn_model, model_args, device)\n                data['logits'].append(logits)\n                pbar.update(1)\n    \n    # Now use the language model server with the custom HF model\n    with language_model_server(model_name, cache_dir, gpu_number):\n        logger.info("Running language model inference...")\n        \n        # Connect to Redis\n        r = redis.Redis(host='localhost', port=6379, db=0)\n        r.flushall()  # Clear all streams\n        \n        # Set up Redis streams\n        remote_lm_input_stream = 'remote_lm_input'\n        remote_lm_output_partial_stream = 'remote_lm_output_partial'\n        remote_lm_output_final_stream = 'remote_lm_output_final'\n        \n        # Initialize timestamps\n        remote_lm_output_partial_lastEntrySeen = get_current_redis_time_ms(r)\n        remote_lm_output_final_lastEntrySeen = get_current_redis_time_ms(r)\n        remote_lm_done_resetting_lastEntrySeen = get_current_redis_time_ms(r)\n        \n        results = {\n            'model_name': model_name,\n            'predictions': [],\n            'true_sentences': [],\n            'sessions': [],\n            'blocks': [],\n            'trials': [],\n        }\n        \n        # Process each trial through the language model\n        with tqdm(total=total_trials, desc=f'LM inference ({model_name})') as pbar:\n            for session in test_data.keys():\n                for trial in range(len(test_data[session]['logits'])):\n                    # Get trial logits and rearrange them\n                    logits = rearrange_speech_logits_pt(test_data[session]['logits'][trial])[0]\n                    \n                    # Reset language model\n                    remote_lm_done_resetting_lastEntrySeen = reset_remote_language_model(\n                        r, remote_lm_done_resetting_lastEntrySeen\n                    )\n                    \n                    # Send logits to LM\n                    remote_lm_output_partial_lastEntrySeen, decoded = send_logits_to_remote_lm(\n                        r,\n                        remote_lm_input_stream,\n                        remote_lm_output_partial_stream,\n                        remote_lm_output_partial_lastEntrySeen,\n                        logits,\n                    )\n                    \n                    # Finalize LM\n                    remote_lm_output_final_lastEntrySeen, lm_out = finalize_remote_lm(\n                        r,\n                        remote_lm_output_final_stream,\n                        remote_lm_output_final_lastEntrySeen,\n                    )\n                    \n                    # Get best prediction\n                    best_sentence = lm_out['candidate_sentences'][0] if lm_out['candidate_sentences'] else ""\n                    \n                    results['predictions'].append(best_sentence)\n                    results['true_sentences'].append(test_data[session]['sentence_label'][trial])\n                    results['sessions'].append(session)\n                    results['blocks'].append(test_data[session]['block_num'][trial])\n                    results['trials'].append(test_data[session]['trial_num'][trial])\n                    \n                    pbar.update(1)\n    \n    # Calculate metrics\n    total_edit_distance = 0\n    total_words = 0\n    \n    for i in range(len(results['predictions'])):\n        true_sentence = remove_punctuation(results['true_sentences'][i]).strip()\n        pred_sentence = remove_punctuation(results['predictions'][i]).strip()\n        \n        ed = editdistance.eval(true_sentence.split(), pred_sentence.split())\n        total_edit_distance += ed\n        total_words += len(true_sentence.split())\n    \n    wer = (total_edit_distance / total_words) * 100 if total_words > 0 else 100.0\n    \n    results.update({\n        'word_error_rate': wer,\n        'total_edit_distance': total_edit_distance,\n        'total_words': total_words,\n        'num_trials': len(results['predictions'])\n    })\n    \n    logger.info(f"Completed evaluation for {model_name}")\n    logger.info(f"Word Error Rate: {wer:.2f}%")\n    \n    return results

def main():\n    parser = argparse.ArgumentParser(description='Test brain-to-text with different HF models using Kaldi integration')\n    parser.add_argument('--models', type=str, required=True,\n                       help='Comma-separated list of HF model names')\n    parser.add_argument('--model_path', type=str, default='data/t15_pretrained_rnn_baseline',\n                       help='Path to pretrained RNN model')\n    parser.add_argument('--data_dir', type=str, default='data/hdf5_data_final',\n                       help='Path to dataset directory')\n    parser.add_argument('--csv_path', type=str, default='data/t15_copyTaskData_description.csv',\n                       help='Path to CSV metadata file')\n    parser.add_argument('--cache_dir', type=str, default='./hf_cache',\n                       help='Directory to cache HF models')\n    parser.add_argument('--gpu_number', type=int, default=0,\n                       help='GPU number to use')\n    parser.add_argument('--output_file', type=str, default='hf_kaldi_comparison.json',\n                       help='Output file for results')\n    \n    args = parser.parse_args()\n    \n    # Set up device\n    if torch.cuda.is_available() and args.gpu_number >= 0:\n        device = torch.device(f'cuda:{args.gpu_number}')\n        logger.info(f'Using GPU: {device}')\n    else:\n        device = torch.device('cpu')\n        logger.info('Using CPU')\n    \n    # Parse model list\n    model_names = [name.strip() for name in args.models.split(',')]\n    logger.info(f"Testing {len(model_names)} models: {model_names}")\n    \n    # Create cache directory\n    os.makedirs(args.cache_dir, exist_ok=True)\n    \n    # Test each model\n    all_results = {}\n    \n    for model_name in model_names:\n        try:\n            logger.info(f"\\n{'='*60}")\n            logger.info(f"Testing model: {model_name}")\n            logger.info(f"{'='*60}")\n            \n            results = evaluate_with_kaldi_integration(\n                model_name=model_name,\n                model_path=args.model_path,\n                data_dir=args.data_dir,\n                csv_path=args.csv_path,\n                device=device,\n                cache_dir=args.cache_dir,\n                gpu_number=args.gpu_number\n            )\n            \n            all_results[model_name] = results\n            \n            # Clear GPU memory\n            if torch.cuda.is_available():\n                torch.cuda.empty_cache()\n                \n        except Exception as e:\n            logger.error(f"Error testing {model_name}: {e}")\n            all_results[model_name] = {"error": str(e)}\n    \n    # Save results\n    with open(args.output_file, 'w') as f:\n        json.dump(all_results, f, indent=2, default=str)\n    \n    # Print summary\n    logger.info(f"\\n{'='*60}")\n    logger.info("SUMMARY")\n    logger.info(f"{'='*60}")\n    \n    for model_name, results in all_results.items():\n        if 'error' in results:\n            logger.info(f"{model_name}: ERROR - {results['error']}")\n        else:\n            wer = results.get('word_error_rate', 'N/A')\n            logger.info(f"{model_name}: WER = {wer:.2f}%")\n    \n    logger.info(f"\\nDetailed results saved to: {args.output_file}")\n\nif __name__ == '__main__':\n    main()
