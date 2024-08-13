# Machine Learning

> **_Work in Progress_**

1. [Definition](#definition)
2. [Loss](#loss)
3. [Accuracy](#accuracy)
4. [Model](#model)
5. [Data Splitting](#data-splitting)
6. [Supervised Learning](#supervised-learning)
7. [Unsupervised Learning](#unsupervised-learning)
8. [Reinforcement Learning](#reinforcement-learning)

## Definition

Machine learning is a type of artificial intelligence that enables computers to learn from data.
It focuses on algorithms without the need of explicit programming.

### Fields

There are three major fields that often overlap.

AI (Artificial intelligence): Enable computers to perform human-like tasks/behaviors  
ML (Machine Learning): Make predictions and solve specific problems using data - subset of AI  
DS (Data Science): Draw insights from data - could use ML

## Loss

The loss is the difference between prediciton and actual label.  
How far is the output from the truth? The smaller the loss, the better performing is the model.

### Functions

$$L_1 = \sum|y_{real} - y_{predicted}|$$
_The further off the prediction, the greater the loss._

$$L_2 = \sum (y_{real} - y_{predicted})^2$$
_Minimal penalty for small misses, much higher loss for bigger ones - quadratic formula._

## Accuracy

Indicates what proportion of the predictions were correct.

## Model

Input -> Model -> Output

Learn about different [models](models.md).

### Input

A.K.A feature/feature vector

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

#### Common File Formats

- CSV: Tabular with header row  
  `id,type,quantity\n0,books,3\n1,pens,5`

- JSON: Tree-like with multiple layers  
  `{[{"id": 0, "type": "books", "quantity": 3}, {"id": 1, "type": "pens", "quantity": 5}]}`

### Output Types

A.K.A prediction

- Classification
  - Multiclass - e.g. Baseball/Basketball/Football
  - Binary - e.g. spam/not
- Regression
  - Continuous values - e.g. Stock price

## Data Splitting

A model isn't trained on all data, because that way you couldn't know how it performs on unseen data. It might perform great on training data, because it has seen it over and over again, but make bad assumptions on new information.  
To assess how well the model can generalize, it is usually split up into 3 datasets:

- Training: Model improves by calculating the loss and learning from it
- Validation: Acts as a reality check, to see if the model can handle unseen data. The loss doesn't get fed back into it.
- Testing: Last check on how the final chosen model performs, based on the loss

## Supervised Learning

Uses inputs with corresponding outputs to train - labeled inputs.

E.g. Picture A is a dog, picture B a cat.

### Training

In order to learn, the labels (results) are extracted from the data.  
Then, each row is fed into the model, which comes up with a prediction.  
This prediction gets compared with the true value. Based on the loss - difference between prediction and actual result - the model makes adjustmens. That's what's called training/learning.

## Unsupervised Learning

Learns about patterns and finds structures to cluster - unlabeled data.

E.g. Picture A, D and E are of have something in common.

## Reinforcement Learning

An agent takes actions in an interactive environment, in order to maximize a reward.
It learns by trial and error and from the feedback (rewards and penalties) it receives.

E.g. This chess move was successfull, maybe use it next time as well.
