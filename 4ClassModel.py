# Import Libraries
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
# Multi-Class Labels Mapping for 5 Classes
multi_class_labels = {
    'Agree': 0,
    'Mostly Agree': 1,
    'Mostly Disagree': 2,
    'Disagree': 3,
    # Handling 'NO MAJORITY': Exclude these rows
}

# Filter out 'NO MAJORITY'
print("Raw values in '5_label_majority_answer':", data['5_label_majority_answer'].unique())
data = data[data['5_label_majority_answer'] != 'NO MAJORITY']

# Map 5-class labels
data['5_label_majority_answer'] = data['5_label_majority_answer'].map(multi_class_labels)

# Debugging: Check data after mapping
print("After mapping:")
print("Unique values in '5_label_majority_answer':", data['5_label_majority_answer'].unique())

# Drop rows with unmapped (NaN) labels
if data['5_label_majority_answer'].isnull().any():
    print(f"Number of NaN values in '5_label_majority_answer': {data['5_label_majority_answer'].isnull().sum()}")
    data = data.dropna(subset=['5_label_majority_answer'])

# Debugging: Check data after dropping NaN
print("After dropping NaN:")
print("Number of rows in dataset:", len(data))
print("Unique values in '5_label_majority_answer':", data['5_label_majority_answer'].unique())

# Check group column 'statement'
if 'statement' not in data.columns:
    raise ValueError("'statement' column is missing in the dataset.")

print("Number of unique statements:", data['statement'].nunique())

# Split 5-class dataset into train and test sets (grouped by statement)
data_multi_5 = data.copy()
multi_5_train, multi_5_test = group_split(data_multi_5, group_col='statement')

# Debugging: Check train and test sizes
print(f"Number of training samples: {len(multi_5_train)}")
print(f"Number of test samples: {len(multi_5_test)}")

# Tokenize data for 5-class classification
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_data(data, tokenizer, max_length=256):
    return tokenizer(
        list(data['tweet']),
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

multi_5_train_encodings = tokenize_data(multi_5_train, tokenizer)
multi_5_test_encodings = tokenize_data(multi_5_test, tokenizer)

# Convert labels to tensors
multi_5_train_labels = torch.tensor(multi_5_train['5_label_majority_answer'].values, dtype=torch.long)
multi_5_test_labels = torch.tensor(multi_5_test['5_label_majority_answer'].values, dtype=torch.long)

# Create PyTorch Dataset for 5-class classification
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

multi_5_train_dataset = TextDataset(multi_5_train_encodings, multi_5_train_labels)
multi_5_test_dataset = TextDataset(multi_5_test_encodings, multi_5_test_labels)

# Model Training and Evaluation Function
def train_and_evaluate_model(train_dataset, test_dataset, num_labels, output_dir):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,  # learning rate (default was 2e-2)
        per_device_train_batch_size=8,  # of batches used in model
        per_device_eval_batch_size=8,
        num_train_epochs=1,  # of total epochs used in model
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        fp16=True  # Enables mixed precision training
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

# Train and Evaluate 5-Class Model
print("Training 5-Class Classification Model...")
train_and_evaluate_model(
    multi_5_train_dataset,
    multi_5_test_dataset,
    num_labels=4,  # Specify 4 classes after excluding "NO MAJORITY"
    output_dir="./multi_5_class_model"
)
