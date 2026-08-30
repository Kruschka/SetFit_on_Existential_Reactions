import pandas as pd
import numpy as np
import transformers


dev_df = pd.read_csv("Development_set.csv")
len(dev_df)
dev_df.columns

# Load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "Alibaba-NLP/gte-base-en-v1.5",
    trust_remote_code=True
)

print(f"Number of posts tokenized: {len(dev_df)}")

token_list = []

for text in dev_df["Text"]:
    # Tokenize the text and get the length of the tokenized input
    tokens = tokenizer.encode(text, add_special_tokens=True)
    token_length = len(tokens)
    token_list.append(token_length)

print(f"Average token length: {np.mean(token_list)}")

token_threshold = 0
# Counting the number of posts exceeding 512 tokens
for t in token_list:
    if t > 512:
        token_threshold += 1

print(f"Number of posts exceeding 512: {token_threshold}")

# Calculating token statistics for all dev posts
max_token = max(token_list)
min_token = min(token_list)
median_token = np.median(token_list)

percentile_90 = np.percentile(token_list, 90)
percentile_95 = np.percentile(token_list, 95)
percentile_99 = np.percentile(token_list, 99)


print(f" Maximum number of tokens in a post: {max_token}")
print(f" Minimum number of tokens in a post: {min_token}")
print(f" Median number of tokens: {median_token}")

print(f" 90th percentile of token lengths: {percentile_90}")
print(f" 95th percentile of token lengths: {percentile_95}")
print(f" 99th percentile of token lengths: {percentile_99}")