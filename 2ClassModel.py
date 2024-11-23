import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# Enable CUDA debugging to get more precise error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # Enable CUDA debugging

# Check GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Function to split dataset by group (statements should not overlap between train/test)
def group_split(data, group_col='statement', test_size=0.2, random_state=42):
    unique_statements = data[group_col].unique()
    train_statements, test_statements = train_test_split(
        unique_statements, test_size=test_size, random_state=random_state
    )
    train_data = data[data[group_col].isin(train_statements)]
    test_data = data[data[group_col].isin(test_statements)]
    return train_data, test_data

# Load Dataset
data = pd.read_csv('Truth_Seeker_Model_Dataset.csv')

# Inspect the dataset
print(data.head())

# Preprocess Data
# Convert 'target' column to string type before applying .str methods
data['target'] = data['target'].astype(str)

# Check for unexpected or invalid string values in the 'target' column before mapping
print("Number of NaN values in 'target' column:", data['target'].isna().sum())
print("Unique values in 'target' column before mapping:", data['target'].unique())

# Clean 'target' column by stripping spaces and converting to lowercase
data['target'] = data['target'].str.strip().str.capitalize()

# Map binary labels
binary_labels = {'True': 0, 'False': 1}
data['target'] = data['target'].map(binary_labels)

# Check for any NaN values after mapping
if data['target'].isna().sum() > 0:
    print("There are NaN values in the target column after mapping. Replacing with 0.")
    data['target'] = data['target'].fillna(0)  # Replace NaN with 0

# Ensure there are no invalid labels
print("Unique labels after mapping:", torch.unique(torch.tensor(data['target'].values)))

# Check if there are any empty strings in the 'tweet' column
data['tweet'] = data['tweet'].fillna('')  # Replace NaN with empty string
empty_tweets = data[data['tweet'].str.strip() == '']  # Filter rows with empty tweets
print(f"Number of empty tweets: {len(empty_tweets)}")

# Filter out empty tweets if desired (optional)
# data = data[data['tweet'].str.strip() != '']

# Split binary and multi-class datasets
data_binary = data.copy()
binary_train, binary_test = group_split(data_binary)

# Tokenize data
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_data(data, tokenizer, max_length=256):
    return tokenizer(
        list(data['tweet']),
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

binary_train_encodings = tokenize_data(binary_train, tokenizer)
binary_test_encodings = tokenize_data(binary_test, tokenizer)

# Convert labels to tensors (ensure labels are of dtype torch.long for CrossEntropyLoss)
binary_train_labels = torch.tensor(binary_train['target'].values, dtype=torch.long)
binary_test_labels = torch.tensor(binary_test['target'].values, dtype=torch.long)

# Check if labels are within the valid range
assert torch.all(binary_train_labels >= 0) and torch.all(binary_train_labels < 2), \
    "Labels for binary classification are not within [0, 1] range."

# Create PyTorch Dataset
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

binary_train_dataset = TextDataset(binary_train_encodings, binary_train_labels)
binary_test_dataset = TextDataset(binary_test_encodings, binary_test_labels)

# Model Training and Evaluation Function
def train_and_evaluate_model(train_dataset, test_dataset, num_labels, output_dir):
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",  # Align evaluation strategy with saving
        save_strategy="epoch",  # Save the model after every epoch
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True  # Load the best model based on evaluation
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()

    # Evaluate Model
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    print("Classification Report:\n", classification_report(test_dataset.labels.numpy(), preds))
    print("Accuracy:", accuracy_score(test_dataset.labels.numpy(), preds))

# Train and Evaluate Binary Model
print("Training Binary Classification Model...")
train_and_evaluate_model(binary_train_dataset, binary_test_dataset, num_labels=2, output_dir="./binary_model")
