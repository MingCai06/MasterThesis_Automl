# MasterThesis_Automl(in processing)

This is a repository of my master thesis.

# AutoML FrameWork
![AutoML FrameWork](https://github.com/MingCai06/_Automl/blob/master/pic/automl_archi.png)


# Requirement
+ Python 3
+ scikit - optimize
+ scikit - learn == 0.20.3
+ lightgbm


# Usage
## Automated Reading (incl. Proprocess)
The data will be read, at the same time, target will be encoded with `LabelEnocoder` and the missing values will be imputed with most frequent value for categorical and mean/median for numerical values.

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

data = reader.read_split(data, target_name=info['target_name'])

# data is a dictionary 
# data = {"train": df_train,
#         "test": df_test,
#         "target": y_train,
#         "y_test": y_test}

```
## Optimization


