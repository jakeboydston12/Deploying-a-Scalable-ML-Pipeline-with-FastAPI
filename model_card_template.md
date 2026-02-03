# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model is a Random Forest used to predict if an individual's income is over $50,000 based on the census data provided.  The model is serialized using Pickle.

## Intended Use

This model is intended to be used as a demonstration of a machine learning pipeline via FastAPI.  It can be used to identify demographic factors correlated with income, though it should not be used in financial or hiring decisions.

## Training Data

This model was trained on the census dataset.  It consists of over 30,000 rows and 14 features of various demographic information.  The training set was created using an 80/20 split of the original data.

## Evaluation Data

Evaluation was performed on the test set consisting of 20% of the total.  The test set used the same column structure and preprocessing as the training set.

## Metrics

The model was evaluated by precision, recall, and F1 score.  These were chosen to help identify high-income individuals.

Precision: 0.7419
Recall: 0.6384
F1: 0.6863

## Ethical Considerations

The dataset includes sensitive attributes such as Race, Sex, and Native Country.  The model could potentially have unfair outcomes if used for automated decision-making.

## Caveats and Recommendations

None