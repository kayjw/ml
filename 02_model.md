# Model

## Basic Concept (Decision Tree)

A real astete eagent might say that he estimates houses by intuition. But on closer inspection, you can see that he recognizes price patterns and uses them to predict new houses.  
Machine learning works the same way.

The Decision Tree is one of the many models.  
It may not be as accurate in predicting as others, but easy to understand and the building block for some of the best models in data science.

This example groups houses into two groups (with price predictions).

        -----------------------> $1.100.000
      /
     / 3 or more
    * bedrooms
     \ less than 3
      \
        -----------------------> $750.000

Training data is then used to fit the model. Which training/adjusting it so that the splits and endprices are as optimal as possible.  
After that, it can be used to predict the prices of new data.

The results of the above tree would be rather vague.  
By using deeper trees (with more splits) you can capture more factors.

    								  ----> $1.300.000
    								/
                                   / yes
          ----------------------- * larger than 11500 square feet lot size
        /                          \ no
       /                            \
      /                               ----> $750.000
     / 3 or more
    * bedrooms
     \ less than 3
      \                               ----> $850.000
       \                            /
        \						   / yes
    	  ----------------------- * larger than 8500 square feet lot size
    						       \ no
    							    \
    								  ----> $400.000

The point at the bottom, where the prediction is made, is called leaf.

## Pandas

Pandas is the main tool for data scientists to explore and manipulate data. Pandas is often abbreviated as `pd`.

```py
import pandas as pd
```

The library has powerful methods for most things that need to be done with data.  
Its most important part is the DataFrame.

### Print Data Summary

```py
data_src = '../input/some-data.csv'
data = pd.read_csv(data_src)
data.describe()
```

Such a table can be interpreted like so:

| Value | Description                                        |
| ----- | -------------------------------------------------- |
| count | Number of non-null objects                         |
| mean  | Average                                            |
| std   | Standard deviation (how numerically spread out)    |
| min   | Minimum value                                      |
| 25%   | First quartile of values (25th percentile)         |
| 50%   | Second quartile of values (50th percentile/median) |
| 75%   | Third quartile of values (75th percentile)         |
| max   | Maximum value                                      |

### Print First 5 Rows

```py
data.head()
```

## Prediction Target

Using the dot notation, you can select the prediction target (column you want to predict).  
This is by convention called y.

```py
y = data.Price
```

## Choosing Features

You could use all columns, except the target, as features. But sometimes you'll be better off with fewer features.  
Those can selected using a feature list.

```py
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = data[features]
```

## Building Model (Decision Tree)

`scikit-learn` is a popular library for modeling the data (typically stored in DataFrames).  
Follow the steps below to create a model.

1. **Define** the type of model and its parameters
2. **Fit** it by capturing patterns from the data
3. **Predict** the results

```py
from sklearn.tree import DecisionTreeRegressor

# 1
model = DecisionTreeRegressor(random_state=1) # random_state for consistent outcome across calls

# 2
model.fit(X, y)

# 3
prediction = model.predict(X)
```

## Validating Model

The fourth step is to evaluate the model by inspecting the prediction accuracy.  
Mean Absolute Error (MAE) is one of many metrics to determine a models quality.  
The formula is simple: `error=actual−predicted`

So the metric shows how much the predictions are off on average.

```py
# 4
mean_absolute_error(y, prediction)
```

## Data Splitting

A model shouldn't be trained on all data, because that way you couldn't know how it performs on unseen data. It might perform great on training data, because it has seen it over and over again, but make bad assumptions on new information.  
To assess how well the model can generalize, it is usually split up into 3 datasets:

- Training: Model improves by calculating the loss and learning from it
- Validation: Acts as a reality check, to see if the model can handle unseen data. The loss doesn't get fed back into it.
- Testing: Last check on how the final chosen model performs, based on the loss

This is how the data can be split up in two pieces:

```py
from sklearn.model_selection import train_test_split


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

model = DecisionTreeRegressor()

# fit using training data
model.fit(train_X, train_y)

# predict validation data
val_predictions = model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
```

## Underfitting and Overfitting

There are two problems that can decrease a models accuracy of future predictions.

- **Overfitting:** So precisely tuned to the training set by capturing patterns that won't recur in the future
- **Underfitting:** Failing to capture relevant patterns

The sweet spot in a decision tree can be found by testing it with different depths:

```py
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

for max_leaf_nodes in [10, 100, 1000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
```

## Random Forest

Decision trees often do not perform as well due to under- or overfitting. Other models face the same problem, but many of those have ideas that can improve performance. One example is the random forest.

A random forest model uses many trees and averages their predictions in order to make a much more accurate prediction than a single tree could.

```py
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

model = RandomForestRegressor(random_state=1)
model.fit(train_X, train_y)
melb_preds = model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
```

## Chosing Model

Using a function you can try out different models.

```py
from sklearn.metrics import mean_absolute_error

def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))
```

## Missing Values

There are multiple approaches to deal with missing values.

| Option                                     | Description                                                                           | Benefit                                    | Disadvantage                                            |
| ------------------------------------------ | ------------------------------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------- |
| Drop                                       | Drop columns with missing values                                                      | Easy to implement                          | Loses access to a lot of potentially useful information |
| Imputation (possibly better than dropping) | Fill in the missing values with some number                                           | Leads to more accurate models              | The imputed value won't be exactly right in most cases  |
| Imputation Extension (standard approach)   | Impute missing values and add a new column to show the location of the imputed values | Will possibly meaningfully improve results | More complex                                            |

### Examples

Don't forget to always adjust both, the training and validation sets/DataFrames.

#### Drop

```py
# get names of cols with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# drop cols those
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("Drop MAE:")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
```

#### Imputation

```py
from sklearn.impute import SimpleImputer

# imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# readd col names (imputation removed them)
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("Imputation MAE:")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
```

#### Imputation with Extension

```py
# make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# make new cols indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# readd col names (imputation removed them)
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("Extensive Imputation MAE:")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))
```

## Categorial Variables

Categorial variables are enums like "Bad, OK, Good, or Great".  
There are three approaches for handling them.

### Drop

Just removing the columns from the dataset is maybe easier but will not work well if the columns contain useful information.

```py
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print("Drop MAE:")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
```

### Ordinal Encoding

This encoding assigns replaces each uniqe value with a different integer, which is a useful approach for ordinal variables:  
Bad (0) < OK (1) < Good (2) < Great (3)

```py
from sklearn.preprocessing import OrdinalEncoder

# copy to avoid changing original data
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# apply ordinal encoder to each col with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

print("Ordinal MAE:")
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
```

### One-Hot Encoding

One-hot encoding creates new columns to represent each unique value in the original data. This encoding typically performs best.

Before:

| Color |
| ----- |
| Black |
| White |
| Blue  |
| White |
| Blue  |

After:

| Black | White | Blue |
| ----- | ----- | ---- |
| 1     | 0     | 0    |
| 0     | 1     | 0    |
| 0     | 0     | 1    |
| 0     | 1     | 0    |
| 0     | 0     | 1    |

```py
from sklearn.preprocessing import OneHotEncoder

# apply one-hot encoder to each col with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# readd index (one-hot encoding removed it)
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# remove categorical cols (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# add one-hot encoded cols to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# ensure all col have string type
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

print("One-Hote Encoding MAE:")
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
```

## Pipelines

Pipelines are a way to chain multiple data transformation and model steps together.  
The data flows through the pipeline and the steps are applied in order.

### Preprocessing Steps

`ColumnTransformer` is a class which is used to bundle together preprocessing steps.  
The example below imputs missing values in numerical, and applies one-hot encoding to categorical data.

```py
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
```

### Model

Of course you need a model to train.

```py
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=0)
```

### Create and Evaluate the Pipeline

Then the `Pipeline` class is used to define a pipeline. Using this pipeline, the preprocessing and fitting can be done in a single line of code, which makes it very readable and easy to use.

```py
from sklearn.metrics import mean_absolute_error

# bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

# preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
```

## Cross-Validation

Cross-validation is a technique where the modeling process is repeated on different subsets of the data.  
The data is split into multiple subsets called "folds". E.g. 4 folds, which hold 25% each of the full data.  
Each of the folds is then used once as the validation, and the other 3 times as the training set.

Advantage: Accurate measure of model quality  
Disadvantage: Takes long to run

Use it for small datasets, which would run for a couple of minutes or less. Where you already have enough data, there is no need to re-use some of it.

```py
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50,
                                                              random_state=0))
                             ])

# -1 since sklearn calculates negative MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE:\n", scores)
```

## XGBoost

Gradient boosting is a very successful algorithm. This method goes through cycles and iteratively adds models to an ensemble.  
The cycle looks like this:

1. An ensemble of models generates predictions for the dataset.
2. Loss function is calculated based on those predictions.
3. A new model gets fit based on the loss function.
4. This new model gets added to the ensemble.
5. Process is repeated.

```py
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

model = XGBRegressor()
model.fit(X_train, y_train)

predictions = model.predict(X_valid)
print("MAE: " + str(mean_absolute_error(predictions, y_valid)))
```

### Parameters

#### n_estimators

Defines cycle amount (be aware of under-/overfitting).

```py
model = XGBRegressor(n_estimators=500)
```

(Typically values from 100-1000.)

#### early_stopping_rounds

Enables the model to find the ideal value for `n_estimators` by stopping early when the scores stop improving.

```py
model.fit(X_train, y_train,
             early_stopping_rounds=10,
             eval_set=[(X_valid, y_valid)],
             verbose=False)
```

This can be combined with a higher `n_estimators` value to find the optimal cycle amount.

#### learning_rate

Predicitons from each model are not just added up, but multiplied by a small number called learning rate.  
Therefore each tree has a smaller effect on the predictions, which can help prevent overfitting.

Small learning rates create more accurate models, but have longer training times due to the higher amount of iterations.

```py
model = XGBRegressor(n_estimators=500, learning_rate=0.05) # default = 0.1
```

#### n_jobs

This paramater has no effect on the resulting model, but it can be used to speed up the training process.
With large data the runtime can be decrased by using parallel processing. On small datasets, this will not have an impact.

```py
model = XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=4)
```

## Data Leakage
