from datasets import load_dataset

# Load IMDb dataset
dataset = load_dataset("imdb")

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Convert to PyTorch
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset.shuffle(seed=42).select(range(20000)),  # subset to speed up
    eval_dataset=test_dataset.select(range(5000)),
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(dim=1).item()
    return "positive" if prediction == 1 else "negative"

print(predict_sentiment("This movie was absolutely fantastic!"))
print(predict_sentiment("It was boring and too long."))

model.save_pretrained("bert-imdb-sentiment")
tokenizer.save_pretrained("bert-imdb-sentiment")
