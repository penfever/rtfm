import nltk
nltk.download('punkt_tab')
import os
import time
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import pandas as pd
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, AutoConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from rtfm.configs import TrainConfig, TokenizerConfig, SerializerConfig
from rtfm.inference_utils import InferenceModel
from rtfm.serialization.serializers import get_serializer
from rtfm.tokenization.text import prepare_tokenizer

train_config = TrainConfig(model_name="mlfoundations/tabula-8b", context_length=8192)

# If using a base llama model (not fine-tuned TabuLa),
# make sure to set add_serializer_tokens=False
# (because we do not want to use special tokens for 
# the base model which is not trained on them).
tokenizer_config = TokenizerConfig()

# Load the configuration
config = AutoConfig.from_pretrained(train_config.model_name)

# Set the torch_dtype to bfloat16 which matches TabuLa train/eval setup
config.torch_dtype = 'bfloat16'

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LlamaForCausalLM.from_pretrained(
    train_config.model_name, device_map="auto", config=config).to(device)

tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
serializer = get_serializer(SerializerConfig())

tokenizer, model = prepare_tokenizer(
    model,
    tokenizer=tokenizer,
    pretrained_model_name_or_path=train_config.model_name,
    model_max_length=train_config.context_length,
    use_fast_tokenizer=tokenizer_config.use_fast_tokenizer,
    serializer_tokens_embed_fn=tokenizer_config.serializer_tokens_embed_fn,
    serializer_tokens=serializer.special_tokens
    if tokenizer_config.add_serializer_tokens
    else None,
)

inference_model = InferenceModel(model=model, tokenizer=tokenizer, serializer=serializer)

DATASETS = {
    'har': 1478, 
    'breast_cancer': 15,
    'semeion': 1501,
    'airlines': 42742
}

all_dataset_results = []

for dataset_name, dataset_id in DATASETS.items():
    start_time = time.time()
    print(f"\n{'='*50}\nFetching dataset: {dataset_name} (ID: {dataset_id})\n{'='*50}")
    
    # Fetch the dataset
    data = fetch_openml(data_id=dataset_id, as_frame=True)
    
    # Extract features and target
    X = data.data
    y = data.target
    # Get the column name if y is a Series
    if isinstance(y, pd.Series):
        y_colname = y.name
    else:
        # If it's a DataFrame, get the first column name
        y_colname = y.columns[0]
    
    # Create train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Select 10 features at random
    X_train = X_train.sample(n=min(10, len(X_train.columns)), random_state=42, axis=1)
    # Select 100 test examples at random
    X_test = X_test[X_train.columns].sample(n=min(100, len(X_test)), random_state=42, axis=0)
    y_test = y_test.loc[X_test.index]
    
    # Create a subset of 8 examples as labeled examples
    labeled_indices = np.random.choice(X_train.shape[0], size=8, replace=False)
    labeled_examples = pd.concat([
        X_train.iloc[labeled_indices].reset_index(drop=True),
        y_train.iloc[labeled_indices].reset_index(drop=True)
    ], axis=1)
    
    # Initialize lists to store results and metrics
    all_results = []
    y_true = []
    y_pred = []
    
    # Process each test example
    for i in range(len(X_test)):
        cur_row = X_test.iloc[i]
        cur_row[y_colname] = y_train.iloc[0]
        target_example = pd.DataFrame(cur_row).T
        
        
        # Get the model prediction
        output = inference_model.predict(
            target_example=target_example,
            target_colname=y_colname,
            target_choices=[str(y_train) for y_train in y_train.unique()],
            labeled_examples=labeled_examples,
        )
        
        # Store results
        all_results.append(output)
        y_true.append(str(y_test.iloc[i]))
        y_pred.append(output)
        print(f"Prediction: {output}")
        print(f"True label: {str(y_test.iloc[i])}")
        
    
    # Calculate metrics
    correct = 0
    total = len(y_true)

    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct += 1
    accuracy = round(correct / total, 2)
    print(f"Accuracy: {accuracy}")
    print(f"Took {time.time() - start_time:.2f} seconds to run inference on {total} examples")
        
    
    # Save results for this dataset
    dataset_results = {
        'dataset_name': dataset_name,
        'dataset_id': dataset_id,
        'accuracy': accuracy,
        'all_predictions': all_results,
        'y_true': y_true,
        'y_pred': y_pred
    }
    all_dataset_results.append(dataset_results)

dataset_results_df = pd.DataFrame(all_dataset_results)
dataset_results_df.to_csv('dataset_results.csv', index=False)
print("Results saved to 'dataset_results.csv'")