import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport

def dataset_analysis(file):
    # Load dataset
    df = pd.read_csv(file)

    profile = ProfileReport(df)
    profile.to_file("static/uploads/output.html")
    
    return profile
