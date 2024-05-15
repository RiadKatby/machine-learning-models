import pandas as pd
from sklearn.datasets import load_diabetes

# Loads the diabetes sample data from sklearn and produces a csv vile that can
# be used by the build/train pipeline script.
def create_sample_data_csv(file_name: str = "diabetes.csv", for_scoring: bool = False):
    sample_data = load_diabetes()
    df = pd.DataFrame(data=sample_data.data, columns:sample_data.feature_names)
    if not for_scoring:
        df['Y'] = sample_data.target
    # Hard code to diabetes so we fail fast if the project has been bootstrapped.
    df.to_csv(file_name, index=False)