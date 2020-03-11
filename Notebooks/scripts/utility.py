"This module retrieves all the components of the selected data type and returns the name, family, and year as a dataframe."
import pandas as pd

def get_component_dataframe(component_type):
    component_type = component_type.lower()
    df = pd.read_pickle(f"./data/preprocessed_data/final_{component_type}_df.pkl")
    if component_type == 'cpu':
        df = df.reset_index()[['name','brand','generation','released']].set_index('name')
    elif component_type == 'gpu':
        df = df.reset_index()[['Release Date','name','brand']].set_index('name')
    else:
        return("Invalid component type entered. Please enter 'GPU' or 'CPU'")
    return df