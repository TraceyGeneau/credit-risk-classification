# Credit Risk Classification

## Overview of the Analysis

The purpose of this model is to assess and predict the cereditworthniness of borrowers using historical lending data from a peer-to-peer lending service.  Through the training and evaluation of this model, informed decisions on lending can be determined minimizing financial risks and ensuring responsible lending practices.  

The lending data model was trained based on loan size, interest rate, borrorwer income, number of accounts, debt to income ratio, derogatory marks, and total debt.  With this information, the modle would predict if a borrower was "Healthy" (0) or "High Risk" (1).  

**Some of the variables I was trying to predict included:**

**Value Counts:** Used to count the number of occurrences of unique values in a categorical variable of series.  It is also useful for understanding if a dataset is balanced or imbalanced.  If the sample set is imbalanced, it is useful to apply oversampling to help balance the model. 

**Test Accuracy:** Used in Machine Learning to understand how well the model is likely to perform on new, unseen data.  It also helps in determining the type of model to use andmodel validation.

**Shape:** Allows for the determination of the (n_samples, n_features) in the dataset.  If the n_samples is equal to 1000 it would indicate there are 1000 rows.  If the n_features is equal to 10 it indicates that there are 10 different points or features for each n_samples.  

**Balanced Accuracy Score:** Assesses the model's performance, particularly when there is a class imbalance.  It measure the model's ability to make accurate predictions while condidering the balance between multiple classes.

**Confusion Matrix:** Is a tabular represtation used to evaluate the performance of a classification model.  It provides a detailed breakdown of the model's predictions compared to the actual true labels.   It breaks down the number of true positive (TP), true negative (TN), false positive (FP), and false negative (FN) predictions for each class in a multi-class or binary classification problem.

example:  ([[TN, FP],
           [FN, TP]]

**Classification Report:** Summary of various evaluation metrics used to assess the performance of a classification model.  
                           Precision:  Accuracy of positive predictions in each class.
                           Recall (Sensitity): Measures the model's ability to correctly identify all instances of a particular class.
                           F1-Score:  The mean of precision and recall.  Helps to assess the trade-off between precision and recall.
                           
**The steps taken for Machine Learning in this analysis included:**

1. Splitting the data into training and testing sets
2. Creating a logistic regression model with the original data set.
3. Pridicting a logistic regression model with resampled training data using RandomOverSampler.
4. Final analysis of the model.

**Logistic Regression**

It is a statistical machine learning algorithm used primarily for binary classes.  Unlike linear regressions, logistic regression employs a logistic (sigmoid shape) function to determine predictions between 0 and 1.  It is best suited for probabilities.  

**Random Over Sampler**

This is a common classification technique used with imbalanced classifications.  It increases the number of instances in the minority class to acheive a more balanced distribution.  It randomly selects instances in the minotiry class and duplicates them to match the number of instances in the majority class.  This helps to fix the imbalance.  

* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:




* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.


## References

Analytics Vidhya. (2022, January 25). Machine Learning Algorithms. Analytics Vidhya. https://www.analyticsvidhya.com/blog/2022/01/machine-learning-algorithms/

Brownlee, J. (n.d.). (2021, July 21). Random Oversampling and Undersampling for Imbalanced Classification. Machine Learning Mastery. https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/