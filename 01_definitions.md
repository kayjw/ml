# Definitions

| Term | Description                             |
| ---- | ----------------------------------------|
| ML   | Machine learning is a type of artificial intelligence that enables computers to learn from data. It focuses on algorithms without the need of explicit programming. |
| **Fields** |
| AI (Artificial Intelligence) | Enable computers to perform human-like tasks/behaviors   |
| ML (Machine Learning) | bla |
| DS (Data Science) | Draw insights from data - could use ML |
| **Learning Types** |
| Supervised Learning | Uses inputs with corresponding outputs to train - labeled inputs. E.g. Picture A is a dog, picture B a cat. |
| Unsupervised Learning | Learns about patterns and finds structures to cluster - unlabeled data. E.g. Picture A, D and E are of have something in common. |
| Reinforcement Learning | An agent takes actions in an interactive environment, in order to maximize a reward. It learns by trial and error and from the feedback (rewards and penalties) it receives. E.g. This chess move was successfull, maybe use it next time as well. |
| **Common File Formats** |
| CSV | Tabular with header row  - `id,type,quantity\n0,books,3\n1,pens,5` |
| JSON | Tree-like with multiple layers - `{[{"id": 0, "type": "books", "quantity": 3}, {"id": 1, "type": "pens", "quantity": 5}]}` |
| **Structure** |
| Process | Input -> Model -> Output |
| **Terms** |
| Input/Feature/Feature Vector |  |
| Model | Function which takes input data and gives a prediction |
| Output/Prediction | Prediction made by the model, based on the input data. |
| Labels/Results | True values associated with input data. |
| Fitting | Process of adjusting the model's parameters to best explain the data. |
| Training/Learning | In order to learn, the labels (results) are extracted from the data. Then, each row is fed into the model, which comes up with a prediction. This prediction gets compared with the true value. Based on the loss - difference between prediction and actual result - the model makes adjustmens. That's what's called training. |
| Training Data | Data used to fit the model |
| Loss | The loss is the difference between prediciton and actual label. How far is the output from the truth? The smaller the loss, the better performing is the model. |
| Accuracy | Indicates what proportion of the predictions were correct. |
| MAE | Mean Absolute Error - average of the absolute differences between predictions and actual values. |
| X | Input data - features. |
| y | Labels/Results - true values associated with input data. |
| X_train | Training data - input data used to fit the model. |
| X_valid | Validation data - input data used to assess the model's performance. |
| y_train | Training labels/Results - true values associated with training data. |
| y_valid | Validation labels/Results - true values associated with validation data. |

## Input Types

- Qualitative: Finite number of categories or groups
  - Nominal data: No inherrent order  
    E.g. Countries
    | Country | One-Hot Encoding |
    | ----------- | ---------------- |
    | Switzerland | [1, 0, 0] |
    | USA | [0, 1, 0] |
    | Italy | [0, 0, 1] |
  - Ordinal data: Inherit order  
    E.g. Age groups  
    Baby is closer to child than adult
- Quantitative: Numerical valued
  - Continuous: Can be measured on a continuum or scale  
    E.g. Temperature
  - Descrete: Result of counting  
    E.g. Number of heads in a sequence of coin tosses

## Output Types

- Classification
  - Multiclass - e.g. Baseball/Basketball/Football
  - Binary - e.g. spam/not
- Regression
  - Continuous values - e.g. Stock price

## Loss Functions

A loss function is a mathematical function that measures the difference between the predicted output and the real output of a model. It is used to quantify the error between the expected and the actual results.  
Here are three commonly used loss functions.

### Mean Absolute Error (MAE)

The further off the prediction, the greater the loss:

$$L_1 = \sum|y_{real} - y_{predicted}|$$


### Mean Squared Error (MSE)

Measures the average squared difference between the predicted and actual values.  
Minimal penalty for small misses, much higher loss for bigger ones:

$$L_2 = \sum(y_{real} - y_{predicted})^2$$

### Cross Entropy Loss

This loss function is used in classification problems where the target variable is categorical. It measures the difference between predicted probabilities and the true probabilities of each class. The formula is:

$$L_3 = - \sum y_{true} \log(y_{predicted}) - (1-y_{true}) \log(1-y_{predicted})$$
