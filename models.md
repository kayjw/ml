# Models

> **_Visit my [Google Colab](https://colab.research.google.com/drive/1YwWtER1868bgalxbipt1o_y0Gcr-tigB?usp=sharing) to see some of the models in action._**

Supervised learning models can be categorized into two main types:

- Classification: Categorial outputs - multiclass or binary
- Regression: Continuous outputs - values

## Classification

### _k_-Nearest Neighbors (kNN)

> **_Assumption:_** Objects that are near each other are similar.

Categorizes a datapoint based on the nearest neighbors - using a distance functions like euclidean, city block and more.

<img src="https://upload.wikimedia.org/wikipedia/commons/7/78/KNN_decision_surface_animation.gif" width="600" alt="KNN decision surface animation" />

_By Paolo Bonfini - Own work, CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=150465667_

### Naive Bayes

> **_Assumption:_** A classes feature is independent of other features.

Classifies based on the highest probability of belonging to a class by calculating the probabilites all features.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/ROC_curves.svg/1920px-ROC_curves.svg.png" width="350" alt="native bayes" style="background-color: white;" />

_By Sharpr for svg version. original work by kakau in a png - Own work, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=44059691_

### More

- Discriminant Analysis

## Regression

### Linear Regression

> **_Assumption:_** Target value is a linear combination of the features.

Uses a linear function for predicting.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Linear_least_squares_example2.svg/1920px-Linear_least_squares_example2.svg.png" width="350" alt="linear regression" />

_By Krishnavedala - Own work, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=15462765_

### More

- Nonlinear Regression
- Generalized Linear Model
- Gaussian Process Regression (GPR)

## Classification or Regression

### Support Vector Machine

> **_Assumption:_** 2 classes can be separated by a divider.

Dataset is divided using a hyperplane, which should linearly separate the classes and have the largest margin between them.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Svm_separating_hyperplanes_%28SVG%29.svg/2560px-Svm_separating_hyperplanes_%28SVG%29.svg.png" width="350" alt="svn" style="background-color: white;" />

_By User:ZackWeinberg, based on PNG version by User:Cyc - This file was derived from: Svm separating hyperplanes.png, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=22877598_

### Neuronal Network

> **_Assumption:_** A structure inspired by the human brain can relate the inputs to desired predictions.

A network consisting of interconnected and layered nodes/neurons are trained by iterative modification of the connection strengths.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Simplified_neural_network_example.svg/2560px-Simplified_neural_network_example.svg.png" width="350" alt="neuronal network" />

_By Mikael Häggström, M.D. Author info- Reusing images- Conflicts of interest:NoneMikael Häggström, M.D. - Own workReference: Ferrie, C., & Kaiser, S. (2019) Neural Networks for Babies, Sourcebooks ISBN: 1492671207., CC0, https://commons.wikimedia.org/w/index.php?curid=137892223_

### More

- Decision Tree
- Ensemble Trees
- Generalized Additive Model (GAM)
