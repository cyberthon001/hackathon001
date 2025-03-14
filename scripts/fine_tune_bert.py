import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# Step 1: Load the Preprocessed Dataset
def load_dataset(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    
    return Dataset.from_dict({
        "text": [item["text"] for item in data],  # Text with Victim Description + Legal Context
        "label": [item["label"] for item in data],  # Numeric label
    })

# Step 2: Preprocess and Tokenize
def preprocess_and_tokenize(dataset, tokenizer):
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
    
    return dataset.map(preprocess_function, batched=True)

# Step 3: Prepare Dataset
def prepare_dataset(tokenized_dataset, tokenizer):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized_dataset, data_collator

# Step 4: Define Model
def define_model(num_labels):
    return AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

# Step 5: Define Training Arguments
def define_training_args():
    return TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,  # Slightly higher learning rate for better adaptation
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,  # Increased epochs for better training
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        fp16=True,  # Mixed precision for speed on GPU
    )

# Step 6: Define Trainer
def define_trainer(model, training_args, train_dataset, eval_dataset, data_collator, tokenizer):
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

# Step 7: Train Model
def train_model(trainer):
    trainer.train()

# Step 8: Save Model
def save_model(model, tokenizer, save_dir):
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

# Main Function
def main():
    # Load the preprocessed dataset
    train_dataset = load_dataset("data/train.json")
    test_dataset = load_dataset("data/test.json")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Preprocess datasets
    tokenized_train = preprocess_and_tokenize(train_dataset, tokenizer)
    tokenized_test = preprocess_and_tokenize(test_dataset, tokenizer)
    
    # Prepare datasets for training
    tokenized_train, data_collator = prepare_dataset(tokenized_train, tokenizer)
    tokenized_test, _ = prepare_dataset(tokenized_test, tokenizer)
    
    # Define model
    num_labels = len(set(train_dataset["label"]))  # Ensure correct number of labels
    model = define_model(num_labels)
    
    # Define training arguments
    training_args = define_training_args()
    
    # Define trainer
    trainer = define_trainer(model, training_args, tokenized_train, tokenized_test, data_collator, tokenizer)
    
    # Train model
    train_model(trainer)
    
    # Save model
    save_model(model, tokenizer, "./fine_tuned_legal_bert")

if __name__ == "__main__":
    main()
