# Credit Risk Classification

## Overview of the Analysis

The purpose of this model is to assess and predict the cereditworthniness of borrowers using historical lending data from a peer-to-peer lending service.  Through the training and evaluation of this model, informed decisions on lending can be determined minimizing financial risks and ensuring responsible lending practices.  

The lending data model was trained based on loan size, interest rate, borrower income, number of accounts, debt to income ratio, derogatory marks, and total debt.  With this information, the model would predict if a borrower was "Healthy" (0) or "High Risk" (1).  

**Some of the variables I was trying to predict included:**

**Value Counts:** Used to count the number of occurrences of unique values in a categorical variable of series.  It is also useful for understanding if a dataset is balanced or imbalanced.  If the sample set is imbalanced, it is useful to apply oversampling to help balance the model. 

**Test Accuracy:** Used in Machine Learning to understand how well the model is likely to perform on new, unseen data.  It also helps in determining the type of model to use and model validation.

**Shape:** Allows for the determination of the (n_samples, n_features) in the dataset.  If the n_samples is equal to 1000 it would indicate there are 1000 rows.  If the n_features is equal to 10 it indicates that there are 10 different points or features for each n_samples.  

**Balanced Accuracy Score:** Assesses the model's performance, particularly when there is a class imbalance.  It measure the model's ability to make accurate predictions while considering the balance between multiple classes.

**Confusion Matrix:** Is a tabular representation used to evaluate the performance of a classification model.  It provides a detailed breakdown of the model's predictions compared to the actual true labels.   It breaks down the number of true positive (TP), true negative (TN), false positive (FP), and false negative (FN) predictions for each class in a multi-class or binary classification problem.

example:  ([[TN, FP],
           [FN, TP]]

**Classification Report:** Summary of various evaluation metrics used to assess the performance of a classification model.  
                           Precision:  Accuracy of positive predictions in each class.
                           Recall (Sensitivity): Measures the model's ability to correctly identify all instances of a particular class.
                           F1-Score:  The mean of precision and recall.  Helps to assess the trade-off between precision and recall.
                           
**The steps taken for Machine Learning in this analysis included:**

1. Splitting the data into training and testing sets
2. Creating a logistic regression model with the original data set.
3. Pridicting a logistic regression model with resampled training data using RandomOverSampler.
4. Final analysis of the model.

**Logistic Regression**

It is a statistical machine learning algorithm used primarily for binary classes.  Unlike linear regressions, logistic regression employs a logistic (sigmoid shape) function to determine predictions between 0 and 1.  It is best suited for probabilities.  

**Random Over Sampler**

This is a common classification technique used with imbalanced classifications.  It increases the number of instances in the minority class to achieve a more balanced distribution.  It randomly selects instances in the minority class and duplicates them to match the number of instances in the majority class.  This helps to fix the imbalance.  

* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

#### Machine Learning Model 1

**Original Confusion Matrix:**

True Positives (TP): 558 - These are instances when the model correctly predicted "High-Risk" loans as "High-Risk."
True-Negatives (TN): 18679 - These are instances when the model correctly predicted "Healthy" loans as "healthy."
False Positives (FP): 67 - These are instances when the model incorrectly predicted "Healthy" loans as "High-Risk."
False Negatives (FN): 80 - These are instances when the model incorrectly predicted "High-Risk" loans as "Healthy."


**Balanced Accuracy** score was 0.944.  Since the value is close to 1 the model is making acurate predictions for both Healthy and High-Risk Groups
        
**Healthy:**   A **precision** score of 1.00 indicates that the model is predicting a healthy loan almost always correctly. **Recall** score of 1.00 indicates that the model is correctly identifying almost all of the healthy loans. **F1 Score** of 1.00 indicates that a balanced mean of precision and recall that indicates an excellent for healthy loans.

**High Risk:**   **Precision** for the high risk loans was 0.87 which means the model predicts the a high risk loans correctly 87% of the time. The **Recall** score was 0.89 meaning that the model predicts about 89% of all the high-risk loans. **F1 Score** of 0.88 indicates that there is a good balance between precision and recall for high-risk loans.

The overall **accuracy** of the model was 0.99 which indicates that the model correctly predicts the labels for 99% of the total samples.

The **Macro average F1 score** is 0.94 which indicates the unweighted average of the F1 scores for both the Healthy and High-Risk Loan. This indicates that the overall performance of the model is excellent.

The **weighted average F1 score** is 0.99 which is a measure of the class imbalance i.e., it considers the class distribution of the model while measuring its performance.

Overall, this regression model can predict both "Healthy" and "High Risk" labels. The precision and recall for healthy loans are extremely high and there is a good overall balance in predicting "High Risk" loans. The F1 score indicates that the model is highly effective at distinguishing between the two classes.


#### Machine Learning Model 2:

**Oversampled Confusion Matrix:**

True Positives (TP): 18521 - These are instances when the model correctly predicted "High Risk" loans as "High Risk."
True Negatives (TN): 18810 - These are instances when the model correctly predicted "Healthy" loans as "healthy."
False Positives (FP): 92 - These are instances when the model incorrectly predicted "Healthy" loans as "High Risk."
False Negatives (FN): 95 - These are instances when the model incorrectly predicted "High Risk" loans as "Healthy."

The **precision, Recall and f1 scores** for both healthy and high-risk groups were either 99% or 100%. These values, combined with the accuracy of 100%, **macro avg** of 100% and **weighted average** of 100% indicated that RandomOverSampler has significantly improved the model's ability to predict both "Healthy" and "High Risk" labels. This suggests the model has become highly effective at distinguishing between the two classes.


## Summary

Both models do very well in predicting the Heathy (0) credit borrowers.  However, the second model created from oversampling, produced a much more precise model when trying to determine High Risk (1) borrowers.  

If the second model is used, there are some considerations that need to be kept in mind:

1. **The risk of overfitting the data:** duplicating the minority samples can lead to the model memorizing the samples rather than learn the right patterns in the data.

2. **Increase in training time**  By oversampling, the dataset size is increased.  This can significantly affect the training time.

3. **Data Leakage:** If oversampling is carried out on the original data set and not the trained data set, there could be a bias of the oversampled data in the model.

4. **Impact on Model Interpretability:** There could be some bias towards the oversampled class and it could make it difficult to interpret the model.  

5. **Data Size:**  Depending on the size of the original dataset, oversampling can lead to really large data sets that might be difficult to work with.  

It would be the most important with credit borrowers to limit the number of False Negatives, incorrectly predicted High-Risk loans.  From the confusion matrix the number of false negatives in the original data was 80 while in the oversampled set it is higher at 95.  Even though the number of false negatives in the oversampled data is higher, it has a slightly higher number of True Positives (TP=18521) compared to the original data set (TP=558).  This indicates that the oversampled model is better at predicting positives or High-Risk cases.  This aligns with the goal is limiting false negatives.  

It would be my suggestion with these two models to proceed with the oversampled data set but to do so with caution.  With the use of real-world data, it is essential to consider overfitting, the reliability of the dataset and the robustness of the model be taking into consideration and monitored.



## References

Analytics Vidhya. (2022, January 25). Machine Learning Algorithms. Analytics Vidhya. https://www.analyticsvidhya.com/blog/2022/01/machine-learning-algorithms/

Brownlee, J. (n.d.). (2021, July 21). Random Oversampling and Undersampling for Imbalanced Classification. Machine Learning Mastery. https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/