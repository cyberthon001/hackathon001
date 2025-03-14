# import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer

# # Load model and tokenizer
# MODEL_PATH = "models/legal_bert"
# tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# def predict(text):
#     """Function to make predictions using the fine-tuned model."""
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#     predicted_class = torch.argmax(probs).item()
#     confidence = probs[0][predicted_class].item()
#     return predicted_class, confidence

# # Run chatbot in terminal
# print("🔹 Cybercrime Chatbot (Type 'exit' to quit)")
# while True:
#     user_input = input("\n👤 You: ")
#     if user_input.lower() == "exit":
#         print("🔹 Chatbot: Goodbye! 👋")
#         break
    
#     predicted_class, confidence = predict(user_input)
#     print(f"🤖 Chatbot: Predicted Category - {predicted_class} (Confidence: {confidence:.2f})")


import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load category mapping from dataset
CATEGORY_MAPPING = {
    0: "Human Trafficking / Forced Labor",
    1: "Online Betting Scam",
    2: "Ransomware Attack",
    3: "Hacking / Unauthorized Access",
    4: "Cyber-Terrorism",
    5: "Online Fraud / Phishing",
    6: "Financial Fraud",
    7: "Identity Theft",
    8: "Cyber Harassment",
    9: "Spam / Scams",
    10: "Data Breach"
}

# Load model and tokenizer
MODEL_PATH = "models/legal_bert"
tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

def predict(text):
    """Function to make predictions using the fine-tuned model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(probs).item()
    confidence = probs[0][predicted_label].item()

    category_name = CATEGORY_MAPPING.get(predicted_label, "Unknown Category")
    return category_name, confidence

# Run chatbot in terminal
print("🔹 Cybercrime Chatbot (Type 'exit' to quit)")
while True:
    user_input = input("\n👤 You: ")
    if user_input.lower() == "exit":
        print("🔹 Chatbot: Goodbye! 👋")
        break
    
    category_name, confidence = predict(user_input)
    confidence_display = f"(Confidence: {confidence:.2f})"
    
    # Display category with confidence warning if it's low
    if confidence < 0.5:
        print(f"⚠️ 🤖 Chatbot: Predicted Category - {category_name} {confidence_display} (Low Confidence, Please Verify)")
    else:
        print(f"🤖 Chatbot: Predicted Category - {category_name} {confidence_display}")
