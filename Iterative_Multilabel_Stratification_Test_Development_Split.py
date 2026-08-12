import numpy as np
import pandas as pd

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# Loading the dataset with the 8 labels
df = pd.read_excel("Reddit_modeling_dataset_8Labels.xlsx")

# Initiating X as the retrieved Reddit-posts
X = df["Text"]

# Define label_columns as the 8 binary label columns
label_columns = [
        "Death Anxiety",
        "Death Acceptance",
        "Loneliness",
        "Solitude",
        "Identity Confusion",
        "Freedom Responsibility",
        "Meaninglessness",
        "Engagement"
]

# Creating the multilabel matrix y consisting of 8 label columns, all posts and and 0/1 as coded values per post
y = df[label_columns].to_numpy()


# Implementing the 90/10 development-test-set split with a single split (dev/test), 
# an assignment of 10% posts to the test set, 
# and a fixed random seed for reproduction purposes
sss = MultilabelStratifiedShuffleSplit(
    n_splits=1, 
    test_size=0.10, 
    random_state=42
)

# Indices for dev and test sets are generated
dev_index, test_index = next(sss.split(X, y))

# Retrieving the datasets
dev_set = df.iloc[dev_index].copy() 
test_set = df.iloc[test_index].copy()

# Checking the sets' number of included posts
print("Development set: ", len(dev_set))
print("Test set: ", len(test_set))

# Print the number of positive instances per label
print("\nFull Dataset positive instances per label: ")
print(df[label_columns].sum())

print("\nDevelopment set positive instances per label:")
print(dev_set[label_columns].sum())

print("\nTest set positive instances per label:")
print(test_set[label_columns].sum())


# Save the newly created Development set and Test set as csv files
dev_set.to_csv("Development_set.csv", index = False)
test_set.to_csv("Test_set.csv", index = False)

