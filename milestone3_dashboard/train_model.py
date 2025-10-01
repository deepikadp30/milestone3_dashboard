from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch
import pandas as pd
from datasets import Dataset

def train_model(df, source_col, target_col, model_name="facebook/bart-base", epochs=3):
    """
    Train a summarization model on user-provided CSV.

    Args:
        df (pd.DataFrame): Input dataframe
        source_col (str): Column name for input text
        target_col (str): Column name for target summary
        model_name (str): Model to fine-tune
        epochs (int): Number of epochs
    """
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Tokenize function (fixed for batch input)
    def preprocess(batch):
        # Ensure batch[source_col] and batch[target_col] are lists of strings
        inputs_text = [str(x) for x in batch[source_col]]
        targets_text = [str(x) for x in batch[target_col]]
        
        inputs = tokenizer(inputs_text, padding="max_length", truncation=True, max_length=512)
        labels = tokenizer(targets_text, padding="max_length", truncation=True, max_length=128)
        inputs["labels"] = labels["input_ids"]
        return inputs
    
    # Convert dataframe to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(preprocess, batched=True)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Training arguments (compatible with older transformers)
    training_args = TrainingArguments(
        output_dir="./trained_model",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        logging_steps=10
        # Removed evaluation_strategy for older versions
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    trainer.train()
    
    # Save the trained model
    model.save_pretrained("./trained_model")
    tokenizer.save_pretrained("./trained_model")
    
    return "./trained_model"
