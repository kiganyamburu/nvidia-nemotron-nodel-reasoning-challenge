"""
Model-based prompt strategy evaluation using Nemotron-3-Nano-30B.
Replaces TF-IDF proxy with actual model.generate() to validate prompt engineering techniques.
"""

import os
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
from pathlib import Path

# ============================================================================
# UTILITY FUNCTIONS (reused from data_analysis.ipynb and prompt_engineering.py)
# ============================================================================

def normalize_answer_text(answer):
    """Normalize answer for comparison: lowercase, strip, handle special chars."""
    if not isinstance(answer, str):
        return ""
    return answer.lower().strip()

def classify_answer_family(answer):
    """Classify answer into family: binary, roman, numeric, text."""
    normalized = normalize_answer_text(answer)
    if re.match(r'^[01]{4,}$', normalized):
        return 'binary'
    elif re.match(r'^[ivxlcdm]+$', normalized):
        return 'roman'
    elif re.match(r'^[-+]?\d+(?:\.\d+)?$', normalized):
        return 'numeric'
    else:
        return 'text'

def classify_problem(prompt):
    """Classify problem type by keyword patterns."""
    prompt_lower = prompt.lower()
    patterns = {
        'Bit Manipulation': [r'\bbit\b', r'\bbinary\b', r'\bshift\b'],
        'Cryptography': [r'\bcipher\b', r'\bencrypt\b', r'\bdecrypt\b', r'\bcode\b'],
        'Logic Puzzles': [r'\bpuzzle\b', r'\blogic\b', r'\briddl\b'],
        'Sequence Analysis': [r'\bsequence\b', r'\bpattern\b', r'\bfibonacci\b'],
        'Mathematical': [r'\bcalculate\b', r'\bsum\b', r'\bmultipl\b', r'\bdivid\b'],
        'Geometry': [r'\bgeometr\b', r'\bshape\b', r'\btriangle\b', r'\bcircle\b'],
        'Graph/Network': [r'\bgraph\b', r'\bnetwork\b', r'\bpath\b', r'\bnode\b'],
        'Geography/Navigation': [r'\bmap\b', r'\bnavigate\b', r'\bdirection\b', r'\blocation\b'],
        'Reasoning': [r'\breason\b', r'\binfer\b', r'\bdeduct\b']
    }
    
    for problem_type, patterns_list in patterns.items():
        for pattern in patterns_list:
            if re.search(pattern, prompt_lower):
                return problem_type
    return 'General'

# ============================================================================
# PROMPT TEMPLATES (same 5 strategies as prompt_engineering.py)
# ============================================================================

def create_prompts(user_prompt, strategy):
    """Create prompt using specified strategy."""
    if strategy == 'minimal':
        return user_prompt
    elif strategy == 'instruction':
        return f"""### Instruction:
{user_prompt}

### Answer:"""
    elif strategy == 'cot':
        return f"""Think step by step:
{user_prompt}

Answer:"""
    elif strategy == 'answer_format':
        return f"""{user_prompt}

Answer (short, exact):"""
    elif strategy == 'few_shot':
        return f"""Example 1: What is 2+2? Answer: 4
Example 2: What is the capital of France? Answer: Paris

Now answer this: {user_prompt}

Answer:"""
    else:
        return user_prompt

# ============================================================================
# MODEL LOADING AND GENERATION
# ============================================================================

def load_model_and_tokenizer(model_name="nvidia/Nemotron-3-Nano-30B-Instruct", 
                              quantize_4bit=True, device_map="auto"):
    """
    Load Nemotron model and tokenizer with optional 4-bit quantization.
    
    Args:
        model_name: Model ID from HuggingFace Hub
        quantize_4bit: If True, use 4-bit quantization for reduced VRAM
        device_map: Device placement strategy ("auto", "cpu", "cuda", etc.)
    
    Returns:
        tokenizer, model, device
    """
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model {model_name}...")
    
    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else None
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else None
        )
    
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")
    
    return tokenizer, model, device

def generate_response(model, tokenizer, prompt, max_new_tokens=64, temperature=0.0):
    """
    Generate a single response from the model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum length of generated tokens
        temperature: Sampling temperature (0.0 = greedy)
    
    Returns:
        Generated text (completion only, without prompt)
    """
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95 if temperature > 0 else 1.0,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and extract only the new tokens (after the prompt)
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = full_text[len(prompt):].strip()
    
    return generated_text

# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def main():
    # Load validation data
    print("Loading validation data...")
    df_train = pd.read_csv("train.csv")
    
    # Use 85/15 split as in prompt_engineering.py
    train_size = int(0.85 * len(df_train))
    df_val = df_train[train_size:].reset_index(drop=True)
    print(f"Validation set size: {len(df_val)}")
    
    # Model selection
    import sys
    use_small_model = "--small" in sys.argv
    
    if use_small_model:
        model_name = "gpt2"  # Tiny model for testing/CPU
        print("Using small model (GPT-2) for testing.")
    else:
        model_name = "nvidia/Nemotron-3-Nano-30B-Instruct"
        print("Using Nemotron-3-Nano-30B for production evaluation.")
    
    # Load model
    try:
        tokenizer, model, device = load_model_and_tokenizer(
            model_name=model_name,
            quantize_4bit=(not use_small_model),  # Only quantize for large models
            device_map="auto"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nFallback options:")
        print("  1. Use --small flag to test with GPT-2 (CPU-compatible)")
        print("  2. Ensure sufficient VRAM (60GB for Nemotron in bfloat16, 15-20GB in 4-bit)")
        print("  3. Check HuggingFace token: huggingface-cli login")
        return
    
    # Define strategies
    strategies = ['minimal', 'instruction', 'cot', 'answer_format', 'few_shot']
    
    # Collect all results
    all_results = []
    
    print("\nRunning generation experiments...")
    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        exact_matches = 0
        total_prompts = len(df_val)
        generated_lengths = []
        
        for idx, row in df_val.iterrows():
            user_prompt = row['prompt']
            gold_answer = normalize_answer_text(row['answer'])
            
            # Create strategy-specific prompt
            full_prompt = create_prompts(user_prompt, strategy)
            
            # Generate response
            generated_answer = generate_response(model, tokenizer, full_prompt)
            generated_answer_normalized = normalize_answer_text(generated_answer)
            
            # Check exact match
            exact_match = (generated_answer_normalized == gold_answer)
            exact_matches += exact_match
            generated_lengths.append(len(generated_answer_normalized))
            
            # Collect result
            all_results.append({
                'strategy': strategy,
                'id': row['id'],
                'prompt': user_prompt,
                'gold_answer': row['answer'],
                'generated_answer': generated_answer,
                'generated_answer_normalized': generated_answer_normalized,
                'exact_match': int(exact_match),
                'similarity': 1.0 if exact_match else 0.0,  # Binary for now; can extend
                'answer_family': classify_answer_family(row['answer']),
                'problem_type': classify_problem(user_prompt),
                'generated_length': len(generated_answer_normalized)
            })
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{total_prompts} prompts")
        
        accuracy = exact_matches / total_prompts
        avg_length = sum(generated_lengths) / len(generated_lengths) if generated_lengths else 0
        print(f"  Exact-match accuracy: {accuracy:.4f} ({exact_matches}/{total_prompts})")
        print(f"  Avg generated length: {avg_length:.1f} chars")
    
    # Save results
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Per-example results
    df_examples = pd.DataFrame(all_results)
    examples_path = output_dir / "model_generation_examples.csv"
    df_examples.to_csv(examples_path, index=False)
    print(f"\nSaved per-example results to {examples_path}")
    
    # Summary by strategy
    summary_data = []
    for strategy in strategies:
        df_strategy = df_examples[df_examples['strategy'] == strategy]
        exact_match_pct = df_strategy['exact_match'].mean()
        avg_length = df_strategy['generated_length'].mean()
        num_tokens = (df_strategy['generated_answer'].apply(
            lambda x: len(tokenizer.tokenize(x))
        ).mean() if len(df_strategy) > 0 else 0)
        
        summary_data.append({
            'strategy': strategy,
            'exact_match': exact_match_pct,
            'num_prompts': len(df_strategy),
            'avg_generated_length': avg_length,
            'avg_num_tokens': num_tokens
        })
    
    df_summary = pd.DataFrame(summary_data)
    summary_path = output_dir / "model_generation_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("MODEL GENERATION EVALUATION SUMMARY")
    print("="*80)
    print(df_summary.to_string(index=False))
    print("="*80)
    
    # Next steps
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Run scripts/compare_proxy_vs_model.py to compare results with TF-IDF proxy")
    print("2. Analyze which strategies improve with real model vs. proxy")
    print("3. Check outputs/comparison_proxy_vs_model.md for detailed insights")
    print("="*80)

if __name__ == "__main__":
    main()
