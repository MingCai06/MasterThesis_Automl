# MasterThesis_Automl

This is a repository of my master thesis.

# AutoML Architecture


# Requirement
+ Python 3
+ scikit - optimize
+ scikit - learn == 0.20.3
+ lightgbm


# Usage
# Automated Reading and Data Cleaning
```python
import reader

# info is a dictionary that contain the description of the input Data.
info = {
    "table_sep": ',',
    "target_name": 'APPETENCY',
    "miss_values": '?'
}

reader = reader.Reader(sep=info['table_sep'],
                       miss_values=info['miss_values'])

data = reader.read_split(["../../data.csv"], target_name=info['target_name'])


```
