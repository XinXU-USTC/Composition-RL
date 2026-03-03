from datasets import Dataset
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Any
import torch
from accelerate import Accelerator
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    StateDictType,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from transformers import Trainer, TrainingArguments


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def preprocess_messages(messages, tokenizer):
    """
    Creates masks for tokens to focus on assistant responses.
    
    Args:
        messages: List of message dictionaries in the chat format
        tokenizer: The tokenizer used for the model
        
    Returns:
        tuple: (text, mask) where mask is 1 for assistant responses and 0 for system/user
    """    
    all_tokens = []
    all_masks = []

    skip_assistant_token = False
    for idx, msg in enumerate(messages):
        if idx == 0:
            add_generation_prompt = True
        else:
            add_generation_prompt = False
            
        msg_text = tokenizer.apply_chat_template(
            [msg], tokenize=False, add_generation_prompt=add_generation_prompt
        )
        
        msg_tokens = tokenizer.encode(msg_text, add_special_tokens=False)

        mask_value = 1 if msg["role"] == "assistant" else 0
        msg_mask = [mask_value] * len(msg_tokens)

        all_tokens.extend(msg_tokens)
        all_masks.extend(msg_mask)
        
    # Add end of sentence token
    eos_token = tokenizer.encode("", add_special_tokens=False)
    all_tokens.extend(eos_token)
    all_masks.extend([1] * len(eos_token))
        
    # Print the decoded tokens that are masked (assistant responses)
    # masked_tokens = [token for token, mask in zip(all_tokens, all_masks) if mask == 1]
    # decoded_masked = tokenizer.decode(masked_tokens)
    # print("Masked tokens (assistant responses):")
    # print(decoded_masked)
    # print("-" * 80)
    # import pdb; pdb.set_trace()

    return all_tokens, all_masks


def find_token_sequence(full_tokens, seq_tokens):
    """Helper function to find a sequence of tokens within a larger sequence"""
    n = len(full_tokens)
    m = len(seq_tokens)
    for i in range(n - m + 1):
        if full_tokens[i:i+m] == seq_tokens:
            return i
    return -1


class ChatDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = 32768  # Set fixed max length to 32k tokens
        self.enable_padding = True
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract messages from each example
        batch_messages = [feature['messages'] for feature in features]
        
        # Process each example
        batch_tokens = []
        batch_attention_masks = []
        batch_labels = []
        
        for messages in batch_messages:
            tokens, loss_mask = preprocess_messages(messages, self.tokenizer)
            tokens = tokens[:self.max_length]  # Truncate to max length
            loss_mask = loss_mask[:self.max_length]
            if not tokens:
                print("Warning: Generated an empty token list for message:", messages)
            batch_tokens.append(tokens)
            attention_mask = [1] * len(tokens)
            batch_attention_masks.append(attention_mask)
            labels = [-100 if mask == 0 else token for token, mask in zip(tokens, loss_mask)]
            batch_labels.append(labels)
            
        # Pad all sequences to fixed max_length (32k)
        if self.enable_padding:
            for i in range(len(batch_tokens)):
                padding_length = self.max_length - len(batch_tokens[i])
                if padding_length > 0:
                    batch_tokens[i].extend([self.tokenizer.pad_token_id] * padding_length)
                    batch_attention_masks[i].extend([0] * padding_length)
                    batch_labels[i].extend([-100] * padding_length)
        
        attention_mask_tensor = torch.tensor(batch_attention_masks)
        print(attention_mask_tensor.shape)
        return {
            "input_ids": torch.tensor(batch_tokens),
            "attention_mask": attention_mask_tensor,
            "labels": torch.tensor(batch_labels)
        }
    

def prepare_training_dataset(data):
    # Convert the data to the format expected by datasets
    dataset_dict = {
        'messages': [item['messages'] for item in data],
    }
    return Dataset.from_dict(dataset_dict).shuffle(42)


def check_sequence_length(tokens, max_length):
    if len(tokens) > max_length:
        print(f"Warning: Sequence length {len(tokens)} exceeds max_length {max_length}")


def main(args):
    # Load and prepare the dataset
    raw_data = load_jsonl(args.data_path)
    dataset = prepare_training_dataset(raw_data)
    
    accelerator = Accelerator(mixed_precision="bf16")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        evaluation_strategy="no",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        bf16=True,
        save_only_model=True,
        weight_decay=0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        deepspeed=args.deepspeed,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1,},
        save_total_limit=None,
    )
    
    with open(args.chat_template, "r") as f:
        template = f.read()
        tokenizer.chat_template = template
    tokenizer.model_max_length = 32768
        
    # Initialize data collator
    data_collator = ChatDataCollator(tokenizer)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Start training
    trainer.train()
    
    # Save the final model
    trainer.save_model(args.model_output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a model with tool call data')
    parser.add_argument('--data_path', type=str, 
                      default='./codeforces_messages.jsonl',
                      help='Path to the JSONL data file')
    parser.add_argument('--model_path', type=str,
                      default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
                      help='Path or name of the pretrained model to fine-tune')
    parser.add_argument('--output_dir', type=str,
                      default='./results',
                      help='Directory for training outputs and checkpoints')
    parser.add_argument('--model_output_dir', type=str,
                      default='./deepcoder-sft',
                      help='Directory to save the final model')
    parser.add_argument('--deepspeed', type=str,
                      default='./config/ds_stage3.json',
                      help='Path to the deepspeed config file')
    parser.add_argument('--chat_template', type=str,
                      default='../rllm/templates/r1-toolcall-python.jinja',
                      help='Path to the chat template file')
    args = parser.parse_args()
   
    main(args)
