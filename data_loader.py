import pandas as pd

def load_dataset():
    data = pd.read_csv(
        "dataset/train_data.txt",
        sep=" ::: ",
        names=["ID", "Title", "Genre", "Description"],
        engine="python"
    )
    return data