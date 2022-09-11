# Report on Data Analysis of the PATH Tabacco Study Series

### John Nesbit, Jacob Aguirre 
### August 18 2022


## 1 Data Processing

We started the Data Processing by downloading the delimitied form of the
Population Assessment of Tobacco and Health (PATH) Study [United States]
Public-Use Files directly from the University of Michigan’s site. After that,
we individually loaded every csv file for each wave of participants, both youth
and adult. We then found every instance of youths starting to smoke, in-
cluding the examples of youths who aged up at the same interval in which
they begin smoking. We found approximately 1820 of these individuals and
added them into a special dataset, using only the collected variables from
the previous wave to that in which they began smoking. We then added an
identical number of youth individuals who did not start smoking the next
interval to that dataset, creating a 50/50 balanced dataset for prediction of
future cigarette usage. In this set aside data, we threw out the variables with
greater than 2000 nan values. We later used a 80/20 train/test data split.
We one-hot encoded the categorical variables and standarized the numerical
variables for the neural network.

## 2 Prediction of Future Cigarette Usage

We implimented two main methods of predicting the future usage of cigarettes
from the individuals in the study. The most effective with an accuracy of
81.12% was a simple three layered deep learning network which was trained
using AdamW, a learning rate of .005, an output activation function of Soft-
max, and Cross Entropy loss on a one hot encoding of the labels of becoming


a smoker or not. The second and less accurate method we used was a gradient
tree boosting algorithm which acheived 80.31% accuracy. The gradient tree
boosting algorithm is extremely important, however, because it has values
which indicate the importance of each value to the algorithm, meaning we
have a sight into what the algorithm is doing as opposed to the black-box-like
uninterpretability of the neural network in this case.

```
Variable Weight
BMI 106
Grade level 55
Parent Education Level 46
Level of Muscle-building Exercise 46
Performance in School 43
```
```
Table 1: importance values for the most heavily weighted variables
```
We did not expect BMI to have such an impact on the model’s predictions,
so we investigated further into what the model might be learning from that
variable by calculating its covariance with other variables. The most covari-
ate variables to BMI were sex, age, relatives’ cholesterol issues, and whether
or not the individual drinks alcohol. These variables seem somewhat reason-
able to be highly covariate with BMI with perhaps the exception of alcohol
usage, but they do not really give us a picture into what the models see in
this variable. In terms of negatively covariate variables, one variable stuck
out: whether the individual had used marijuana in the past. This variable,
though confusing as to why there is a covariance with BMI, provides insight
as to what the model may be seeing within the BMI variable as marijuana
usage is extremely correlated with cigar and cigarette usage in this dataset.

### 2.1 Deep Learning Network Overfit

We observed that while learning, my initial network would quickly acheive
92% percent train accuracy with a test accuracy of only 80.22% percent and
then improves no more as the gradient signals proveded by the 8% of incor-
rectly labeled data was not enough to drag the network out of its training
minima into a more general minima. We reduced the size of the network
moving from an inital hidden size 2x that of the data dimesionality to.

![Acc](https://user-images.githubusercontent.com/49009243/189551873-507623c8-3b58-4555-bb53-0fbd5f71fd7b.png)

Figure 1: A Graph of Model’s Accuracy as the Models Train

of the data dimensionality. Interestingly, these efforts increased the lag be-
tween train and test accuracy, moving the final train accuracy to 99.89% and
the test accuracy to 77%. Thankfully, we were granted relief by adding 2
droupout layers with a p value of .5. This addition changed the train accu-
racy to 91.69% and the test accuracy to 81.12%. The accuracy graph shows
an extended period of stagnation in the model without dropout from epoch
125-150, which adding dropout eliminates.
The following graph shows the effects of adding dropout on the gradi-
ent signal sent through the 3rd layer of the neural network. Before adding
dropout, the signal is so weak that it is unlikely to make any changes to
the accuracy of the system, even during a AdamW restart. After adding
dropout, the signal nearly doubles and the network has a better chance of
avoiding a false local minima.

![Figure_1](https://user-images.githubusercontent.com/49009243/189551902-d6f8309f-f2aa-4d4e-a7db-a9f49878ad51.png)

Figure 2: A graph of Gradient Signal as the Model Train


## 3 Acknowlegments

We want to thank the Georgia Tech Department of Economics for funding the creation of this report
