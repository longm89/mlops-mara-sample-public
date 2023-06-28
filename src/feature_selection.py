import argparse


import pandas as pd
import numpy as np
from optbinning import OptimalBinning
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

class FeatureSelector:
    @staticmethod
    def filter_quasi_constant(train_x, num_cols, cat_cols):
        quasi_constant_feat = []

        # iterate over every feature
        for feature in num_cols + cat_cols:
            predominant = (train_x[feature].value_counts() / np.float(
                len(train_x))).sort_values(ascending=False).values[0]

            # evaluate predominant feature: do more than 99% of the observations
            # show 1 value?
            if predominant > 0.99:
                quasi_constant_feat.append(feature)
        return quasi_constant_feat
    
    @staticmethod
    def calculate_psi(col1, col2):
        # Create a union of unique values from col1 and col2 as bins
        unique_values = pd.Series(pd.concat([col1, col2])).unique()
        
        # Calculate the percentage in each bin for col1 and col2
        prop_col1 = col1.value_counts(normalize=True).reindex(unique_values, fill_value=0)
        prop_col2 = col2.value_counts(normalize=True).reindex(unique_values, fill_value=0)
        
        # Add a small value to avoid division by zero
        prop_col1 = prop_col1 + 1e-10
        prop_col2 = prop_col2 + 1e-10
        
        # Calculate the PSI
        psi = np.sum((prop_col1 - prop_col2) * np.log(prop_col1 / prop_col2))
        
        return psi
    
    @staticmethod
    def filter_psi(train_x, train_y, test, num_cols, cat_cols, target_col, threshold=0.25):
        train_temp = train_x.copy()
        test_temp = test.copy()
        # WoE transform the train_temp and test_temp of num_cols
        for col in num_cols:
            optb = OptimalBinning(name=col, dtype="numerical", solver="cp")
            optb.fit(train_temp[col], train_y[target_col].values)
            train_temp[col] = optb.transform(train_temp[col], metric = "woe")
            test_temp[col] = optb.transform(test_temp[col], metric = "woe")

        psi_dict = {}
        for col in num_cols + cat_cols:
            psi_dict[col] = FeatureSelector.calculate_psi(train_temp[col], test_temp[col])
        
        exc_cols = []
        for col in psi_dict:
            if psi_dict[col] > threshold:
                exc_cols.append(col)
        return exc_cols
    
    @staticmethod
    def feature_importance_random_forest(train_x, train_y):
        rf = RandomForestClassifier() 
        rf.fit(train_x, train_y)
        importances = rf.feature_importances_
        # Sort the feature importances in descending order
        sorted_indices = np.argsort(importances)[::-1]

        # Set a threshold to select the top k features (e.g., keep features with importance above 0.01)
        threshold = 0.01

        # Get the indices of the features to keep
        selected_feature_indices = sorted_indices[importances[sorted_indices] > threshold]
        selected_feature_names = train_x.columns[selected_feature_indices]

        return selected_feature_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

