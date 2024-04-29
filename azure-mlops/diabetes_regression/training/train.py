import os
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def split_data(df):
    X = df.drop('Y', axis=1).values
    y = df['Y'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    data = { "train": {"X": X_train, "y": y_train},
        "test": {"X": X_test, "y": y_test}}

    return data
    
def train_model(data, ridge_args):
    reg_model = Ridge(**ridge_args)
    reg_model.fit(data["train"]["X"], data["train"]["y"])
    return reg_model

def get_model_metrics(model, data):
    preds = model.predict(data["test"]["X"])
    mse = mean_squared_error(preds, data["test"]["y"])
    metrics = {"mse": mse}
    return metrics

def main():
    print("Running train.py")

    # define training parameters
    ridge_args = {"alpha": 0.5}

    # load the training data a dataframe
    data_dir = "data"
    data_file = os.path.join(data_dir, 'diabetes.csv')
    train_df = pd.read_csv(data_file)

    data = split_data(train_df)

    # Train the model
    model = train_model(data, ridge_args)

    # Log the metrics for the model
    metrics = get_model_metrics(model, data)
    for (k, v) in metrics.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()