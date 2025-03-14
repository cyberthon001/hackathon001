import json
from sklearn.model_selection import train_test_split

# Load the dataset
with open("data/Final_Extended_Cybercrime_Dataset.json", "r") as f:
    data = json.load(f)

# Create a mapping from string labels to integers
categories = [
    "Online Cyber Trafficking",
    "Online Gambling or Betting",
    "Ransomware",
    "Cryptocurrency Crime",
    "Cyber Terrorism",
    "Online and Social Media Related Crime",
    "Online Financial Fraud",
    "Hacking and Damage to Computer Systems",
    "Child/Women-Related Crimes"
]
label_to_id = {label: idx for idx, label in enumerate(categories)}

# Prepare the dataset for text classification
formatted_data = []
for entry in data:
    formatted_data.append({
        "text": entry["Victim Description"],
        "label": int(label_to_id[entry["Category"]])  # Convert int64 to int
    })

# Split the dataset into train, test, and validation sets
train_data, test_data = train_test_split(formatted_data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

# Save the splits
with open("data/train.json", "w") as f:
    json.dump(train_data, f, indent=4)  # Added indent for better readability
with open("data/test.json", "w") as f:
    json.dump(test_data, f, indent=4)
with open("data/val.json", "w") as f:
    json.dump(val_data, f, indent=4)

print("Dataset preprocessing complete!")
