# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Spam Detection using Linear Regression
#
# This notebook demonstrates how to build a spam detection system using linear regression.
# We'll use text feature extraction techniques and logistic regression (which is a linear model)
# to classify emails as spam or not spam.

# %% [markdown]
# ## Import Required Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
import kagglehub
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# %% [markdown]
# ## Load Real Spam Email Dataset
#
# We'll use a "real" spam email dataset from Kaggle to make our analysis more realistic and robust.
# This dataset contains actual spam and ham (legitimate) emails.

# %%
# Download the spam email dataset from Kaggle
print("Downloading spam email dataset from Kaggle...")
path = kagglehub.dataset_download("jackksoncsie/spam-email-dataset")
print("Path to dataset files:", path)

# %%
# Load the dataset
import os
# List files in the dataset directory
dataset_files = os.listdir(path)
print("Files in dataset:", dataset_files)

# Load the CSV file 
dataset_file = csv_files[0]  # Take the first CSV file
df = pd.read_csv(os.path.join(path, dataset_file))
print(f"\nLoaded dataset: {dataset_file}")
print(f"Dataset shape: {df.shape}")
print(f"Column names: {list(df.columns)}")
print("\nSome random rows:")
print(df.sample(20))

# %%
# Rename columns for consistency
df = df.rename(columns={text_col: 'message', label_col: 'label'})

# Check unique values in label column
print(f"\nUnique labels: {df['label'].unique()}")
print(f"Label counts:\n{df['label'].value_counts()}")

# %%
# Standardize labels to 0 (ham) and 1 (spam)
# Handle different label formats
if df['label'].dtype == 'object':
    # If labels are strings, map them to numbers
    unique_labels = df['label'].unique()
    print(f"Original labels: {unique_labels}")
    
    # Common mappings
    label_mapping = {}
    for label in unique_labels:
        label_str = str(label).lower()
        if 'spam' in label_str or label_str in ['1', '1.0']:
            label_mapping[label] = 1
        elif 'ham' in label_str or 'legitimate' in label_str or label_str in ['0', '0.0']:
            label_mapping[label] = 0
        else:
            # If unclear, assume first unique value is ham (0), second is spam (1)
            if label == unique_labels[0]:
                label_mapping[label] = 0
            else:
                label_mapping[label] = 1
    
    print(f"Label mapping: {label_mapping}")
    df['label'] = df['label'].map(label_mapping)

# Ensure labels are integers
df['label'] = df['label'].astype(int)

# Remove any rows with missing values
df = df.dropna(subset=['message', 'label'])

print(f"\nFinal dataset shape: {df.shape}")
print(f"Spam messages: {df[df['label'] == 1].shape[0]}")
print(f"Ham messages: {df[df['label'] == 0].shape[0]}")
print(f"Spam ratio: {df['label'].mean():.3f}")

# Show sample messages
print(f"\nSample spam messages:")
spam_samples = df[df['label'] == 1]['message'].head(3)
for i, msg in enumerate(spam_samples, 1):
    print(f"{i}. {msg[:100]}..." if len(msg) > 100 else f"{i}. {msg}")

print(f"\nSample ham messages:")
ham_samples = df[df['label'] == 0]['message'].head(3)
for i, msg in enumerate(ham_samples, 1):
    print(f"{i}. {msg[:100]}..." if len(msg) > 100 else f"{i}. {msg}")

# %% [markdown]
# ## Exploratory Data Analysis

# %%
# Display first few messages
print("Sample messages:")
print("\nSpam messages:")
print(df[df['label'] == 1]['message'].head(3).values)
print("\nHam messages:")
print(df[df['label'] == 0]['message'].head(3).values)

# %% [markdown]
# ## Understanding Bag of Words (BoW)
#
# Before we build our model, let's understand how **Bag of Words** works step by step.
# Bag of Words is a fundamental text representation technique that converts text into numerical vectors.

# %% [markdown]
# ### Step 1: Building the Vocabulary
#
# First, let's manually walk through how bag of words is constructed using a few sample messages.

# %%
# Sample messages for demonstration
demo_messages = [
    "Free money now!",
    "Meeting at noon",
    "Free lunch offer",
    "Money back guarantee"
]

print("Demo messages:")
for i, msg in enumerate(demo_messages):
    print(f"{i+1}. '{msg}'")

# Step 1: Tokenization - split into words
print("\nStep 1: Tokenization")
print("=" * 30)
tokenized_messages = []
for i, msg in enumerate(demo_messages):
    tokens = msg.lower().split()  # Simple tokenization
    tokenized_messages.append(tokens)
    print(f"Message {i+1}: {tokens}")

# Step 2: Build vocabulary (unique words)
print("\nStep 2: Building Vocabulary")
print("=" * 30)
vocabulary = set()
for tokens in tokenized_messages:
    vocabulary.update(tokens)

vocabulary = sorted(list(vocabulary))  # Sort for consistent ordering
print(f"Vocabulary: {vocabulary}")
print(f"Vocabulary size: {len(vocabulary)}")

# Step 3: Create word-to-index mapping
print("\nStep 3: Word-to-Index Mapping")
print("=" * 30)
word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
for word, idx in word_to_idx.items():
    print(f"'{word}' -> index {idx}")

# %%
# Step 4: Convert messages to numerical vectors
print("Step 4: Converting Messages to Vectors")
print("=" * 40)

bow_matrix = []
for i, tokens in enumerate(tokenized_messages):
    # Initialize vector with zeros
    vector = [0] * len(vocabulary)
    
    # Count occurrences of each word
    for token in tokens:
        if token in word_to_idx:
            idx = word_to_idx[token]
            vector[idx] += 1
    
    bow_matrix.append(vector)
    print(f"\nMessage {i+1}: '{demo_messages[i]}'")
    print(f"Tokens: {tokens}")
    print(f"Vector: {vector}")
    
    # Show which positions correspond to which words
    non_zero_positions = [(idx, count) for idx, count in enumerate(vector) if count > 0]
    print(f"Non-zero positions: {[(vocabulary[idx], count) for idx, count in non_zero_positions]}")

# %%
# Visualize the Bag of Words matrix
import pandas as pd

bow_df = pd.DataFrame(bow_matrix, columns=vocabulary)
bow_df.index = [f"Msg {i+1}" for i in range(len(demo_messages))]

print("\nBag of Words Matrix:")
print("=" * 50)
print(bow_df)

# Visualize as heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(bow_df, annot=True, cmap='Blues', fmt='d', cbar_kws={'label': 'Word Count'})
plt.title('Bag of Words Matrix Visualization\n(Rows = Messages, Columns = Words)')
plt.xlabel('Words in Vocabulary')
plt.ylabel('Messages')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Key Insights from the Manual BoW Construction:
#
# 1. **Vocabulary Creation**: We collect all unique words across all documents
# 2. **Vector Size**: Each document becomes a vector of size = vocabulary size
# 3. **Word Counts**: Each position in the vector represents the count of that word
# 4. **Sparsity**: Most positions are 0 (words don't appear in most documents)
# 5. **Order Independence**: "free money" and "money free" will wind up as the same.
#
# Now let's use scikit-learn's CountVectorizer to do this automatically for our spam detection task. CountVectorizer is an internal tool for implementing bag of words.

# %% [markdown]
# ## Data Preprocessing and Feature Engineering
#
# We'll use **Bag of Words (CountVectorizer)** to convert text messages
# into numerical features that our linear regression model can understand.

# %%
# Split the data
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Training set spam ratio: {y_train.mean():.2f}")
print(f"Test set spam ratio: {y_test.mean():.2f}")

# %% [markdown]
# ## Model Training
#
# We'll use a pipeline that combines Bag of Words vectorization with Logistic Regression.
# Logistic Regression is a linear model that's well-suited for binary classification tasks.

# %%
# Create and train the model pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer(
        max_features=1000,  # Limit to top 1000 features
        stop_words='english',  # Remove common English stop words
        lowercase=True,  # Convert to lowercase
        ngram_range=(1, 2)  # Use both unigrams and bigrams
    )),
    ('classifier', LogisticRegression(
        random_state=42,
        max_iter=1000
    ))
])

# Train the model
print("Training the model...")
pipeline.fit(X_train, y_train)
print("Model training completed!")

# %% [markdown]
# ## Examining the Bag of Words Matrix for Our Dataset
#
# Let's look at how our actual spam/ham messages are converted to bag of words vectors.

# %%
import random

# Get the fitted vectorizer and examine a random sample
bow_vectorizer = pipeline.named_steps['bow']
feature_names = bow_vectorizer.get_feature_names_out()

print(f"Vocabulary size: {len(feature_names)}")
print(f"Sample of 20 random words: {random.sample(list(feature_names), 20)}")

# %% [markdown]
# ### Understanding N-grams in Bag of Words
#
# Notice that our vocabulary contains both single words (unigrams) and two-word phrases (bigrams).
# This is because we set `ngram_range=(1, 2)` in our CountVectorizer.
#
# - **Unigrams (1-gram)**: Single words like "free", "money", "click"
# - **Bigrams (2-gram)**: Two-word phrases like "free money", "click here", "amazing deals"
#
# Let's separate and examine these different types of features:

# %%
# Separate unigrams and bigrams
unigrams = [word for word in feature_names if ' ' not in word]
bigrams = [word for word in feature_names if ' ' in word]

print(f"Total features: {len(feature_names)}")
print(f"Unigrams (single words): {len(unigrams)}")
print(f"Bigrams (two-word phrases): {len(bigrams)}")

print(f"\Random 20 unigrams: {random.sample(list(unigrams), 20)}")
print(f"\Random 20 bigrams: {random.sample(list(bigrams), 20)}")

print(f"\nWhy use bigrams?")
print("- Captures phrases like 'free money' which is more spam-indicative than just 'free' or 'money' alone")
print("- Helps with context: 'account closed' vs just 'account' or 'closed'")
print("- Can improve classification performance by capturing common spam phrases")

# %%
# Visualize the bag of words matrix for a subset of messages
# Select a few messages for visualization
sample_indices = [0, 1, 5, 6, 10, 15]  # Mix of spam and ham
sample_msgs = [df.iloc[i]['message'] for i in sample_indices]
sample_lbls = [df.iloc[i]['label'] for i in sample_indices]

# Transform these messages
bow_matrix = bow_vectorizer.transform(sample_msgs)

# Convert to dense array and create DataFrame
bow_dense = bow_matrix.toarray()

# Only show features that appear in at least one of these messages
feature_mask = bow_dense.sum(axis=0) > 0
active_features = feature_names[feature_mask]
active_bow_matrix = bow_dense[:, feature_mask]

# Create DataFrame for visualization
bow_viz_df = pd.DataFrame(
    active_bow_matrix,
    columns=active_features,
    index=[f"{'SPAM' if lbl else 'HAM'} {i+1}" for i, lbl in enumerate(sample_lbls)]
)

print(f"\nBag of Words Matrix for {len(sample_msgs)} sample messages:")
print(f"Showing {len(active_features)} features that appear in these messages")
print("=" * 70)

# Show the matrix
if len(active_features) <= 20:  # If few features, show all
    print(bow_viz_df)
    
    # Visualize as heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(bow_viz_df, annot=True, cmap='Blues', fmt='d', cbar_kws={'label': 'Word Count'})
    plt.title('Bag of Words Matrix for Sample Messages\n(Rows = Messages, Columns = Words)')
    plt.xlabel('Words')
    plt.ylabel('Messages')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:  # If many features, show top ones
    # Show features with highest variance (most discriminative)
    feature_variance = bow_viz_df.var(axis=0)
    top_features = feature_variance.nlargest(15).index
    
    print(bow_viz_df[top_features])
    
    # Visualize as heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(bow_viz_df[top_features], annot=True, cmap='Blues', fmt='d', cbar_kws={'label': 'Word Count'})
    plt.title('Bag of Words Matrix for Sample Messages\n(Top 15 Most Variable Features)')
    plt.xlabel('Words')
    plt.ylabel('Messages')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Model Evaluation

# %%
# Make predictions
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.3f}")
print(f"AUC Score: {auc_score:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# %% [markdown]
# ## Visualization of Results

# %%
# Create visualization plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xticklabels(['Ham', 'Spam'])
axes[0, 0].set_yticklabels(['Ham', 'Spam'])

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0, 1].set_xlim([0.0, 1.0])
axes[0, 1].set_ylim([0.0, 1.05])
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].legend(loc="lower right")

# Feature Importance (Top Bag of Words features)
feature_names = pipeline.named_steps['bow'].get_feature_names_out()
coefficients = pipeline.named_steps['classifier'].coef_[0]

# Get top 10 positive and negative coefficients
top_positive_idx = np.argsort(coefficients)[-10:]
top_negative_idx = np.argsort(coefficients)[:10]

top_features = np.concatenate([top_negative_idx, top_positive_idx])
top_coeffs = coefficients[top_features]
top_feature_names = [feature_names[i] for i in top_features]

colors = ['red' if coeff < 0 else 'blue' for coeff in top_coeffs]
axes[1, 0].barh(range(len(top_coeffs)), top_coeffs, color=colors, alpha=0.7)
axes[1, 0].set_yticks(range(len(top_coeffs)))
axes[1, 0].set_yticklabels(top_feature_names)
axes[1, 0].set_xlabel('Coefficient Value')
axes[1, 0].set_title('Top 20 Features (Red=Ham, Blue=Spam)')

# Prediction probabilities distribution
axes[1, 1].hist(y_pred_proba[y_test == 0], alpha=0.7, label='Ham', bins=10, density=True)
axes[1, 1].hist(y_pred_proba[y_test == 1], alpha=0.7, label='Spam', bins=10, density=True)
axes[1, 1].set_xlabel('Spam Probability')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Distribution of Prediction Probabilities')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Testing with New Messages

# %%
# Test the model with some new messages
test_messages = [
    "Congratulations! You've won a free iPhone! Click here to claim!",
    "Hey, can we meet for lunch tomorrow?",
    "URGENT: Your bank account needs verification!",
    "The meeting is scheduled for 2 PM in conference room A",
    "Get rich quick with this amazing opportunity!"
]

print("Testing with new messages:")
print("=" * 50)

for i, message in enumerate(test_messages, 1):
    prediction = pipeline.predict([message])[0]
    probability = pipeline.predict_proba([message])[0]
    
    label = "SPAM" if prediction == 1 else "HAM"
    spam_prob = probability[1]
    
    print(f"\nMessage {i}: {message}")
    print(f"Prediction: {label}")
    print(f"Spam Probability: {spam_prob:.3f}")

# %% [markdown]
# ## Model Interpretation
#
# Let's examine what the model learned by looking at the most important features.

# %%
# Get feature importance
feature_names = pipeline.named_steps['bow'].get_feature_names_out()
coefficients = pipeline.named_steps['classifier'].coef_[0]

# Create a DataFrame for easier analysis
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients
})

# Sort by coefficient value
feature_importance = feature_importance.sort_values('coefficient', key=abs, ascending=False)

print("Top 15 Most Important Features:")
print("=" * 40)
print("Positive coefficients indicate spam-like features")
print("Negative coefficients indicate ham-like features")
print()

for idx, row in feature_importance.head(15).iterrows():
    direction = "SPAM" if row['coefficient'] > 0 else "HAM"
    print(f"{row['feature']:20} | {row['coefficient']:8.3f} | {direction}")

# %% [markdown]
# ## Summary and Conclusions
#
# In this notebook, we successfully built a spam detection system using logistic regression. However, can you see why this approach might fail if I were to deploy it and not retrain it frequently?

# %%
print("Analysis complete! 🎉")
print(f"Final model accuracy: {accuracy:.1%}")
print(f"Final model AUC score: {auc_score:.3f}")

# %%
