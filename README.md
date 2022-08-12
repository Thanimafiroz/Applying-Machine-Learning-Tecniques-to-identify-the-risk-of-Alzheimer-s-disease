# Applying-Machine-Learning-Tecniques-to-identify-the-risk-of-Alzheimer-s-disease
Abstract
In recent years,there has been a significant uptick in interest in Alzheimer’s disease(AD), a form 
of dementia characterized by a gradual and deadly accumulation of proteins in the brain, killing 
brain cells and leading to memory loss,behavior changes,and eventually death. While there is 
currently no cure for Alzheimer’s disease, research has suggested that there is a correlation 
between the risk of contracting Alzheimer’s and various aspects of the brain, including grey matter 
distribution, appearance of amyloid deposits etc.
Problem Statement
Thanks to the improvement in medical imaging technology,namely magnetic resonance 
imaging(MRI) and positron emission tomography(PET) scans,the doctors and researchers are now 
able to diagnose Alzheimer’s disease with greater accuracy and timeliness,however, such scans 
often only yield telltale information after symptoms are reported,at which point measures to slow 
or prevent its development are usually of limited effecteveness.While there are of course many 
factors that can lead to Alzheimer’s,and not all can be determined from brain scans alone,I have 
considered the socio economic factors,gender as well as education as independent predictors.
MRI findings include both, local and generalized shrinkage of brain tissue. Below is a pictorial 
representation of tissue shrinkage:
Some studies have suggested that MRI features may predict rate of decline of AD and may guide 
therapy in the future.However in order to reach that stage clinicians and researchers will have to 
make use of machine learning techniques that can accurately predict progress of a patient from 
mild cognitive impairment to dementia.
In this project,I am trying to develop a sound machine learning model that can help doctors to 
predict early Alzheimer’s using MRI data.
Here I tend to do a comparative study of different commonly used machine learning 
classification algorithms like logical regression, Decision Tree, Random Forest, and Support 
Vector machine. Choosing the best machine learning algorithm in order to solve the problems of 
classification and prediction of data is the most important part of machine learning which 
depends on the dataset as well. 
Related Works
A variety of other projects regarding classification of Alzheimer’s disease using brain scan 
imagery have been conducted in recent years. The original publication has only done some 
preliminary exploration of the MRI data as majority of their work was focused towards data 
gathering. However, in the recent past there have been multiple efforts that have been made to 
detect early-alzheimers using MRI data. Some of the work that was found in the literature was as 
follows:
1) Machine learning framework for early MRI-based Alzheimer's conversion prediction in 
MCI subjects-In this paper the authors were interested in identifying mild cognitive 
impairment(MCI) as a transitional stage between age-related coginitive decline and 
Alzheimer's. The group proposes a novel MRI-based biomaker that they developed using 
machine learning techniques. They used data available from the Alzheimer's Disease 
Neuroimaging Initiative ADNI Database.
2) Detection of subjects and brain regions related to Alzheimer's disease using 3D MRI 
scans based on eigenbrain and machine learning-The authors of this paper have proposed 
a novel computer-aided diagnosis (CAD) system for MRI images of brains based on 
eigenbrains and machine learning.
3) Support vector machine-based classification of Alzheimer’s disease from whole-brain 
anatomical MRI-In this paper the authors propose a new method to discriminate patients 
with AD from elderly controls based on support vector machine (SVM) classification of 
whole-brain anatomical MRI. The authors used three-dimensional T1-weighted MRI 
images from 16 patients with AD and 22 elderly controls and parcellated them into 
regions of interests (ROIs). They then used a SVM algorithm to classify subjects based 
upon the gray matter characteristics of these ROIs. Based on their results the classifier 
obtained 94.5% mean correctness.
The above-mentioned 3 papers over here have explored the same question. Regardless, it is 
worthwhile to mention that the above papers were exploring raw MRI data and on the other 
hand, in this project I am dealing with 3 to 4 biomarkers that are generated from MRI images.
Dataset Description
The Oasis-3 dataset includes a plethora of metadata in addition to patient scans,going as specific 
as family history and number of psychiatric appointments.
This project represents the datasets for training and predictions,both featuring MRI scan data 
from normal patients to patients having Alzheimer’s. The dataset I am using is Oasis-3.This 
dataset is a longitudinal study consisting of 1,099 images taken from roughly 150 subjects,with 
each subject undergoing a minimum of three scan sessions,each atleast a year apart.Of the 150 
subjects,72 showed no symptoms of Alzheimer’s disease over the course of the study.64 showed 
symptoms of Alzheimer’s disease prior to any scans and remains so throughout the course of all 
scans and 14 did not initially display any symptoms of Alzheimer’s disease but did after 
subsequent scans.The Oasis-3 dataset relies upon PET scans which reveal brain functions.The 
subject’s aged from 60-96and everyone is right-handed.
Column Description
Column names Full-form
EDUC Years of Education
SES Socio economic status
MMSE Mini Mental State Examination
CDR Clinical Dementia Rating
E TIV Estimated Total Intracranial Volume
nWBV Normalize whole brain volume
ASF Atlas Scaling Factor
Data Preprocessing
The below list of data preprocessing has been done on the dataset.
i. Considered first visit data only as I am trying to analyze the risk of occurrence of 
Alzheimer’s disease.
ii. Performed one hot coding in the male/female column.
iii. Changed the target variable column name to ‘Group’ and performed one hot coding after
that.
iv. Dropped the unwanted columns like ‘MRI ID’,’ visit’, and ’hand’(since all are righthanders).
v. Handling Missing Values- There are 8 rows with missing values in the ‘SES’ column.
These missing values can be handled in two ways-
• Dropping missing rows
• Imputation-replacing the missing values with the corresponding values. Since the 
data size is small, I assume imputation would help to improve the performance of 
the model.
Exploratory Data Analysis
In this section, I have focused on exploring the relationship between each feature of MRI tests 
and the dementia of the patient. The reason for conducting this EDA is to state the relationship 
of data explicitly through a graph so that the correlations can be assumed before data extraction 
or data analysis. It might help to understand the nature of the data and to select the appropriate 
analysis method for the model later.
The minimum, maximum, and average values of each feature for graph implementation are as 
follows.
Col name Min Max Mean
Educ 6 23 14.6
SES 1 5 2.34
MMSE 17 30 27.2
CDR 0 1 0.29
eTIV 1123 1989 1490
nWBV 0.66 0.837 0.73
ASF 0.883 1.563 1.2
Below conclusions obtained after EDA.
• Men are more likely with demented, an Alzheimer's Disease, than Women.
• Demented patients were less educated in terms of years of education.
• Nondemented group has higher brain volume than Demented group.
• Higher concentration of 70-80 years old in Demented group than those in the 
nondemented patients.
Learning Methods
As mentioned previously, the algorithms chosen to implement the Alzheimer’s prediction model
are Logistic Regression, Support Vector Machine, Random Forest, and Decision Tree.
1) Logistic Regression Model
I choose logistic regression as my base model because it is one of the simplest model for 
a classification task.The Logistic Regression Model assumes the presence of Alzheimer’s 
given feature model as a Bernoulli RV:
 y|x; θ ~ Bernoulli(η)
As part of the exponential family, under the assumption that the natural parameter η is 
linearly related to the input η= θ
T x, the probability equation is as below.
P(y=1)|x: θ=hθ(x)=g(θ
T x)=1/(1+e- θx
).
Logistic Regression is a binary classifier, meaning Alzheimer’s is said to be 
present(y=1)when hθ(x)>=0.5,else y=0.
For the logistic regression model, I have considered the dataset with dropping missing 
values as baseline model and obtained below results.
Similarly, Logistic regression has been done with the model with imputation and 
obtained the following results.
Overall, the dataset with imputation outperforms the one without imputation. Hence, for 
the later models, dataset with imputation is considered.
2) Support Vector Machine
Support vector machine (SVM) is considered one of the best algorithms for 
supervised learning. The main idea of this algorithm is to map the data from a
relatively low dimensional space to a relatively high dimensional space so that the 
higher dimensional data can be separated into two classes by a hyperplane. The 
hyperplane that separates the data with maximum margin is called the support 
vector classifier, which be determined using Kernel Functions in order to avoid 
expensive computation to transform the data explicitly.
The setting of SVM in this report:
• C: Penalty parameter C of the error term. [0.001, 0.01, 0.1, 1, 10, 100, 
1000]
• gamma: kernel coefficient. [0.001, 0.01, 0.1, 1, 10, 100, 1000]
• kernel: kernel type. ['rbf', 'linear', 'poly', 'sigmoid']
3) Decision Tree Methods
• Random Forest Classifier
Tree classification is very powerful to classify the nonlinear datasets. The 
classification includes bagged tree, random forest, and boosting . Random 
forest provides an improvement over the bagged trees. Bagged trees 
consider all the predictors 
(p predictors) in every split of the tree, whereas random forest limits the 
selection of the predictors to m predictors. The number of predictors 
considered in the split in random forest is equal to the square root of the 
total number of predictors,m=sqrt(p). In other words, random forest 
decorrelates the trees through considering less predictors. Unlike highly 
correlated bagged trees, the variance in random forest is significantly 
decreased . The setting of random forest in this report:
n_estimators(M): the number of trees in the forest-2
 max_features(d): the number of features to consider when looking for the best 
split-5
max_depth(m): the maximum depth of the tree-7
K is the class number. M is the sample size. 
The value will take on a small value if the 
node is pure.
• Boosting Classifier
Boosting the classifier is another approach to tree classification. Boosting 
also becomes a method to improve the predictions over bagged trees. 
Boosting trees are grown sequentially. Each tree is grown based on the 
information from previously grown trees, thus robust to overfitting. 
Notably, boosting does not involve bootstrap sampling; instead, each tree 
collectively fits on the original tree. The setting of boosting in this report:
• The number of boosting trees: 2
• Test criterion: MSE. 
• Learning rate: 0.0001
Results and Discussion
The performance matrix of each model is as follows.
Below is a comparison of our results with those from the papers that were listed previously.
SL 
No
Paper Data Model Results
1 E. Moradi et al Ye et al Random 
Forest 
Classifier
AUC=71.0% ACC=55.3%
Filipovych et al. Random 
Forest 
Classifier
AUC=61.0% ACC=N/A
Zhang et al. Random 
Forest 
Classifier
AUC=94.6% ACC=N/A
Batmanghelich et 
al.
Random 
Forest 
Classifier
AUC=61.5% ACC=N/A
2 Zhang et al. Ardekani et al. Support 
Vector 
Machine
Polynomial 
Kernal
AUC=N/A ACC=92.4%
Linear Kernal AUC=N/A ACC=91.5%
Radial Basis 
Function
AUC=N/A ACC=86.7%
3 Hyun, Kyuri, 
Saurin
Marcus et al. Logistic 
Regression 
with 
imputation
AUC=79.2% ACC=78.9%
Logistic 
Regression 
with dropna
AUC=70.0% ACC=78.9%
Support 
Vector 
Machine
AUC=82.2% ACC=75.0%
Decision Tree 
Classifier
AUC=82.5% ACC=81.6%
Random 
Forest 
Classifier
AUC=84.4% ACC=84.2%
Adaboost AUC=82.5% ACC=84.2%
It can be noticed that my results are comparable and in certain cases better than those from the 
previous work. Our Random Forest Classifier was one of the best performing model.
Unique Approach
The uniqueness of this model is the fact that I have included metrics like MMSE and Education 
also to train inorder to differentiate between normal healthy adults and those with Alzheimer's. 
MMSE is one of the gold standards for determining dementia and hence therefore it is an 
important feature to include.
The same fact also make this approach flexible enough to be applied to other neurodegenerative 
diseases which are diagnosed using a combination of MRI features and cognitive tests.
Limitations
There are limitations in implementing a complex model because of the quantity of the dataset. 
Even though the nature of each feature is evident, the ranges of each group's test value are not 
classified well. In other words, I could identify more clearly the differences in the variables 
which might have played a role in the result. The predicted value using the random forest model 
is higher than the other models. It implies there is a potential for a higher prediction rate if we 
pay more attention to developing the data cleaning and analysis process. Moreover, the perfect 
recall score of 1.0 of SVM 1.0 indicates that the quality and accuracy of the classification might 
decrease dramatically when we use different datasets
