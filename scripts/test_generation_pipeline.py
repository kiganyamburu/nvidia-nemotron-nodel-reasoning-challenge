"""
Testing script for model generation evaluation pipeline.
Uses GPT-2 (tiny, CPU-friendly) to demonstrate the full workflow without requiring large GPU.
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import re

def normalize_answer_text(answer):
    """Normalize answer for comparison."""
    if not isinstance(answer, str):
        return ""
    return answer.lower().strip()

def classify_problem(prompt):
    """Quick problem classification."""
    prompt_lower = prompt.lower()
    if 'calculate' in prompt_lower or 'sum' in prompt_lower or 'multiply' in prompt_lower:
        return 'Mathematical'
    elif 'cipher' in prompt_lower or 'encrypt' in prompt_lower:
        return 'Cryptography'
    else:
        return 'General'

def test_generation_pipeline():
    """Test model generation pipeline with GPT-2."""
    
    print("Testing model generation pipeline with GPT-2...")
    print("="*80)
    
    # Load tiny model
    print("Loading GPT-2 tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}\n")
    
    # Create sample prompts
    sample_prompts = [
        "What is 2 + 2?",
        "Calculate the sum of 5, 10, and 15.",
        "What is the capital of France?",
    ]
    
    print("Sample generation tests:")
    print("-" * 80)
    
    for i, prompt in enumerate(sample_prompts, 1):
        print(f"\n[Test {i}] Prompt: {prompt}")
        
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=20,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated_text[len(prompt):].strip()
        
        print(f"  Generated: {completion[:60]}...")
    
    print("\n" + "="*80)
    print("Pipeline test completed successfully!")
    print("="*80)
    
    # Now test with actual data
    print("\nTesting with actual validation data...")
    
    df_train = pd.read_csv("train.csv")
    train_size = int(0.85 * len(df_train))
    df_val = df_train[train_size:].reset_index(drop=True)
    
    # Test on first 5 validation examples
    results = []
    for idx in range(min(5, len(df_val))):
        row = df_val.iloc[idx]
        prompt = row['prompt']
        gold_answer = normalize_answer_text(row['answer'])
        
        # Simple prompt strategy
        full_prompt = f"Question: {prompt}\nAnswer:"
        
        inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=20,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated[len(full_prompt):].strip()
        
        results.append({
            'id': row['id'],
            'prompt': prompt[:50] + "...",
            'gold_answer': gold_answer,
            'generated': completion[:30] + "...",
            'problem_type': classify_problem(prompt)
        })
    
    df_results = pd.DataFrame(results)
    print("\nFirst 5 validation results:")
    print(df_results.to_string(index=False))
    
    print("\n" + "="*80)
    print("✓ Pipeline test successful!")
    print("Ready to run full model_generation_evaluation.py with Nemotron or other target model.")
    print("="*80)

if __name__ == "__main__":
    test_generation_pipeline()
