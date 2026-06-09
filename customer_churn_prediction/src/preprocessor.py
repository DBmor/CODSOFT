from sklearn.preprocessing import LabelEncoder

class Preprocessor:
    def preprocess(self, df):

        # Remove unnecessary columns
        df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

        encoder = LabelEncoder()

        for col in df.select_dtypes(include='object'):
            df[col] = encoder.fit_transform(df[col])
        print(df.columns)
        X = df.drop("Exited", axis=1)
        y = df["Exited"]

        return X, y