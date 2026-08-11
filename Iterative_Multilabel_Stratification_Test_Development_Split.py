import numpy as np
import pandas as pd

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# Loading the dataset with the 8 labels
df = pd.read_excel("C:\Users\rebec\OneDrive\Dokumente\UT 25_26\Master_Thesis\Reddit_modeling_dataset_8Labels.xlsx")

# Initiating X as the retrieved Reddit-posts
X = df["Text"]

# Initiating y as the 8 binary label columns
y = df[
        "Death Anxiety",
        "Death Acceptance",
        "Loneliness",
        "Solitude",
        "Identity Confusion",
        "Freedom Responsibility",
        "Meaninglessness",
        "Engagement"
]

y = df[label_columns].to_numpy()



sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=0)
sss.get_n_splits()

print(sss)



class sklearn.model_selection.StratifiedShuffleSplit()

    for i, train_index, test_index) in enumerate(sss.split(X, y))
        print 