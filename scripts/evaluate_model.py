import json
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the test dataset
with open("data/test.json", "r") as f:
    test_data = json.load(f)

# Convert to Hugging Face Dataset
test_dataset = Dataset.from_list(test_data)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_legal_bert")
model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_legal_bert")

# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# Ensure correct format
tokenized_test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load trainer
trainer = Trainer(model=model)

# Evaluate to get eval_loss
eval_results = trainer.evaluate(tokenized_test_dataset)

# Get predictions
predictions = trainer.predict(tokenized_test_dataset)

# Extract logits and convert to predicted labels
logits = predictions.predictions
pred_labels = np.argmax(logits, axis=1)  # Convert logits to class labels
true_labels = predictions.label_ids  # True labels

# Calculate accuracy, precision, recall, F1-score
accuracy = accuracy_score(true_labels, pred_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="weighted")

# Print results
print(f"Evaluation Results:")
print(f"- Loss: {eval_results['eval_loss']:.4f}")  # Fetching eval_loss from correct variable
print(f"- Accuracy: {accuracy:.4f}")
print(f"- Precision: {precision:.4f}")
print(f"- Recall: {recall:.4f}")
print(f"- F1-score: {f1:.4f}")
