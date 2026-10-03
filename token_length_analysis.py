import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

token_array = np.array(token_list)

plot_df = pd.DataFrame(token_array, columns=["Token Length per Post"])

fig, ax = plt.subplots(figsize=(9, 4.5))

# Creating the violin plot
violin_plot = sns.violinplot(
    data=plot_df, 
    x="Token Length per Post", 
    inner=None,
    ax=ax,
    color="lightblue"
)

# Adding individual posts with slight vertical jitter
np.random.seed(42)
y_jitter = np.random.normal(0, 0.03, size=len(token_array))

ax.scatter(
    token_array, 
    y_jitter, 
    color="blue", 
    s=8,
    alpha=0.2
)

# Adding reference lines for sequence length limits
ax.axvline(512, linestyle="--", linewidth=1.2, color="purple", label="BGE Maximum Sequence Length (512 Tokens)")
ax.axvline(1536, linestyle="dashdot", linewidth=1.2, color="orange", label="GTE Sequence Length Used (1,536 Tokens)")
ax.axvline(median_token, linestyle="dotted", linewidth=1, color="gray", label=f"Median Token Length ({int(median_token)} Tokens)")

ax.legend(loc="upper right", fontsize=11, frameon=True, facecolor="white", edgecolor="gray")
ax.tick_params(axis="x", labelsize=11)

ax.set_xlabel("Token Length per Post", fontsize=11, labelpad=8)
ax.set_ylabel("")
ax.set_yticks([])

fig.tight_layout()


# Save the violin plot as a PNG file and a PDF file 
fig.savefig("Violin_Plot.png", dpi = 300, bbox_inches = "tight") 
fig.savefig("Violin_Plot.pdf", bbox_inches = "tight")

plt.show()

