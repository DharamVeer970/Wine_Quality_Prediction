# Wine Quality Prediction

A machine-learning project that predicts whether a red wine is **good** or
**bad** from its physicochemical properties. It uses the classic
[Wine Quality (red)](https://archive.ics.uci.edu/dataset/186/wine+quality)
dataset - 1,599 Portuguese "Vinho Verde" red wines - and trains a Random Forest
classifier that reaches about **82% accuracy**.

The whole workflow lives in a single Jupyter notebook,
[`WineQuality.ipynb`](WineQuality.ipynb): load the data, explore how each
chemical property relates to quality, reframe the problem as binary
classification, balance the classes, and tune a model with grid search.

## The dataset

Each row is one wine sample with 11 numeric input features and a `quality`
score from 3 to 8 (assigned by human tasters):

| Feature                | What it measures                                  |
|------------------------|---------------------------------------------------|
| fixed acidity          | Non-volatile acids (tartaric)                     |
| volatile acidity       | Acetic acid - high levels taste like vinegar      |
| citric acid            | Adds freshness and flavour                        |
| residual sugar         | Sugar left after fermentation                     |
| chlorides              | Salt content                                      |
| free sulfur dioxide    | Free SO2, protects against microbes               |
| total sulfur dioxide   | Free + bound SO2                                   |
| density                | Density relative to water                         |
| pH                     | Acidity on the 0-14 scale                         |
| sulphates              | Additive that contributes to SO2                  |
| alcohol                | Alcohol percentage                                |
| **quality**            | Target - taster score from 3 to 8                 |

The raw scores are imbalanced: most wines are a 5 or 6, with very few at the
extremes (only 10 wines scored 3, and 18 scored 8).

## Exploratory analysis

Before modelling, the notebook plots every feature against the quality score to
see which properties actually move with quality:

![Feature vs quality bar charts](output.jpg)

Each panel is one feature, with quality (3-8) along the x-axis and the feature's
mean on the y-axis; the black bars are confidence intervals. A few clear
relationships stand out:

- **Alcohol** rises steadily with quality - the strongest positive signal.
- **Sulphates** and **citric acid** also trend upward in better wines.
- **Volatile acidity** falls as quality rises - too much of it (the vinegar
  taste) marks a wine down.
- **Chlorides** (saltiness) edge down in higher-quality wines.
- **Density** and **pH** are almost flat across quality, so they carry little
  signal on their own.

These trends are what a good model should pick up on.

## Approach

1. **Binarize the target.** The 3-8 score is bucketed into two classes:
   scores of 6.5 and below become **bad** (`0`), above become **good** (`1`).
   This turns a hard six-way problem into a cleaner yes/no one.
2. **Balance the classes.** After bucketing there are 1,382 bad wines but only
   217 good ones. The majority class is randomly down-sampled to 217 so the
   model sees a 50/50 split and can't win by always guessing "bad".
3. **Split.** 70% train / 30% test.
4. **Train and tune.** A `RandomForestClassifier` is tuned with `GridSearchCV`
   over the number of trees (`n_estimators` from 100 to 1000), using 10-fold
   cross-validation scored on accuracy.
5. **Evaluate.** Accuracy, a confusion matrix, and a classification report on
   the held-out test set.

## Results

The tuned Random Forest scores about **0.82 accuracy** on the test set, with
balanced precision and recall across both classes:

```
              precision    recall  f1-score   support
           0       0.81      0.81      0.81        62
           1       0.83      0.83      0.83        69
    accuracy                           0.82       131
```

Exact numbers shift slightly between runs because the majority-class
down-sampling is random and not seeded, but accuracy lands around 0.82 each
time.

## Tech stack

| Tool             | Role                                          |
|------------------|-----------------------------------------------|
| pandas / numpy   | Data loading and manipulation                 |
| seaborn / matplotlib | Exploratory plots                         |
| scikit-learn     | Train/test split, Random Forest, grid search  |
| Jupyter          | Notebook environment                          |

## Getting started

### 1. Clone

```bash
git clone https://github.com/DharamVeer970/Wine_Quality_Prediction.git
cd Wine_Quality_Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

```bash
jupyter notebook WineQuality.ipynb
```

Run the cells top to bottom. Or execute it headlessly:

```bash
jupyter nbconvert --to notebook --execute WineQuality.ipynb
```

## Project structure

```
Wine_Quality_Prediction/
|-- WineQuality.ipynb       # The full analysis and model
|-- winequality_red.csv     # Dataset (1,599 red wines)
|-- output.jpg              # Feature-vs-quality bar charts
|-- requirements.txt        # Python dependencies
`-- README.md
```

## Possible improvements

- Set a random seed on the down-sampling so results are reproducible.
- Try keeping all the data with class weights instead of down-sampling, so no
  rows are thrown away.
- Compare other models (the notebook already imports SVM and SGD) and report
  which wins.
- Add feature-importance output to confirm alcohol and sulphates drive the
  prediction, as the exploratory plots suggest.
