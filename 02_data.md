# Data

## Pandas

Pandas is the main tool for data scientists to explore and manipulate data. Pandas is often abbreviated as `pd`.

```py
import pandas as pd
```

The library has powerful methods for most things that need to be done with data.  
Its most important part is the DataFrame.

```py
data_src = '../input/some-data.csv'
data = pd.read_csv(data_src)
data.describe() # prints a summary of the data
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

## Data Splitting

A model isn't trained on all data, because that way you couldn't know how it performs on unseen data. It might perform great on training data, because it has seen it over and over again, but make bad assumptions on new information.  
To assess how well the model can generalize, it is usually split up into 3 datasets:

- Training: Model improves by calculating the loss and learning from it
- Validation: Acts as a reality check, to see if the model can handle unseen data. The loss doesn't get fed back into it.
- Testing: Last check on how the final chosen model performs, based on the loss
