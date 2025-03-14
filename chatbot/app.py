from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ✅ Correct Model Path
MODEL_PATH = "fine_tuned_legal_bert"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Define categories (replace with your actual labels)
categories = [
    "Online Cyber Trafficking",
    "Online Gambling",
    "Ransomware",
    "Cryptocurrency Crime",
    "Cyber Terrorism",
    "Social Media Crime",
    "Online Fraud",
    "Hacking",
    "Child/Women-Related Crimes"
]

app = FastAPI()

class UserInput(BaseModel):
    text: str

@app.post("/predict")
def predict_category(user_input: UserInput):
    inputs = tokenizer(user_input.text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1).item()

    return {
        "category": categories[predicted_label],
        "confidence": round(float(torch.max(probabilities).item()), 4)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
