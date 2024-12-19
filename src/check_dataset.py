import pandas as pd


# NOTE: this check is the case of ESC-50 dataset.
def check():
    meta_df = pd.read_csv("data/meta/esc50.csv")
    print(meta_df)
    categories = meta_df["category"].unique()
    print(f"num of dataset: {meta_df.shape[0]}")
    print(len(categories), categories[:10])


if __name__ == "__main__":
    check()
