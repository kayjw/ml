# Machine Learning

## Definition

Machine learning is a type of artificial intelligence that enables computers to learn from data.
It focuses on algorithms without the need of explicit programming,

### Fields

There are three major fields that often overlap.

AI (Artificial intelligence): Enable computers to perform human-like tasks/behaviors  
ML (Machine Learning): Make predictions and solve specific problems using data - subset of AI  
DS (Data Science): Draw insights from data - could use ML

### Loss

The loss is the difference between prediciton and actual label.  
How far is the output from the truth? The smaller the loss, the better performing is the model.

#### Functions

$$L_1 = \sum|y_{real} - y_{predicted}|$$
_The further off the prediction, the greater the loss._

## Supervised Learning

Uses inputs with corresponding outputs to train - labeled inputs.

E.g. Picture A is a dog, picture B a cat.

### Model

Input -> Model -> Output

_Input is also called "feature vector" and output is known as "prediction"._

### Input Types

A.K.A Features

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

### Output Types

A.K.A Prediction

- Classification
  - Multiclass - e.g. Baseball/Basketball/Football
  - Binary - e.g. spam/not
- Regression
  - Continuous values - e.g. Stock price

### Training

In order to learn the labels (results), are extracted from the data.  
Then, each row is fed into the model, which comes up with a prediction.  
This prediction gets compared with the true value. Based on the loss - difference between prediction and actual result - the model makes adjustmens. That's what's called training/learning.

#### Data

A model isn't trained on all data, because that way you couldn't know how it performs on unseen data. It might perform great on training data, because it has seen it over and over again, but make bad assumptions on new information.  
To assess how well the model can generalize, it is usually split up into 3 datasets:

- Training: Model improves by calculating the loss and learning from it
- Validation: Acts as a reality check, to see if the model can handle unseen data. The loss doesn't get fed back into it.
- Testing: Last check on how the final chosen model performs, based on the loss

## Unsupervised Learning

Learns about patterns and finds structures to cluster - unlabeled data.

E.g. Picture A, D and E are of have something in common.

## Reinforcement Learning

An agent takes actions in an interactive environment, in order to maximize a reward.
It learns by trial and error and from the feedback (rewards and penalties) it receives.

E.g. This chess move was successfull, maybe use it next time as well.
