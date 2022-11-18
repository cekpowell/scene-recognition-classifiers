# Scene Recognition Classifiers
## COMP3204: Computer Vision
---
## Contents

- **[Introduction](#introduction)**
  * **[Task Description](#task-description)**
  * **[Project Contents](#project-contents)**
- **[Running the Application](#running-the-application)**
- **[Usage](#usage)**
  * **[Performing Image Classification](#performing-image-classification)**

---

## Introduction

### Task Description

- As a group, use the [**Open-IMAJ**](http://openimaj.org/) library to develop three scene recognition image classifiers.
- The specification of the classifiers is as follows:
  - **Classifier 1** : A [**K-Nearest Neighbour Classifier**](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm#:~:text=An%20object%20is%20classified%20by,of%20that%20single%20nearest%20neighbor.) using the [**Tiny Image**](https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj4/html/euzun3/index.html#:~:text=The%20%22tiny%20image%22%20feature%2C,zero%20mean%20and%20unit%20length.) feature as the feature vector for each image.
  - **Classifier 2** : A set of 15 one-vs-all [**Linear Classifiers**](https://en.wikipedia.org/wiki/Linear_classifier#:~:text=A%20linear%20classifier%20achieves%20this,vector%20called%20a%20feature%20vector.) using a [**Bag-of-Visual-Words**](https://towardsdatascience.com/bag-of-visual-words-in-a-nutshell-9ceea97ce0fb) feature based on fixed size densely-sampled pixel patches.
  - **Classifier 3** : The **best possible Classifier** the group can implement (i.e., free to make whatever classifier desired with the goal of maximising performance.)
    - After trialing three different classifiers, the team chose a [**Naive Bayes Classifier**](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) that uses a **Pyramid Histogram of Words** feature vector based on [**Dense SIFT features**](http://www.scholarpedia.org/article/Scale_Invariant_Feature_Transform).
- Each of these classifiers must then be trained on provided training data and used to classify provided testing data.
  - *The application of each classifier to the testing data is referred to as a **Run** (i.e., classifier 1 = run 1, classifier 2 = run 2, ...)* 

### Project Contents

#### Source Code

- **Three directories**:
  - `Run1` : Contains the implementation of the first classifier (K-Nearest Neighbour Classifier). 
  - `Run2` : Contains the implementation of the second classifier (Linear Classifiers).
  - `Run3` : Contains the implementation of the third classifier (best possible classifier).
  - Each directory contains one file for the classifier used in the run, and one file for the feature extractor.
-  **The following `java` files**:
  - `App.java` : Program used to evaluate the performance of each classifier, and run it on the provided testing data.
  - `MyClassifier.java` and `Tuple.java` : Helper classes for the classifier implementations.

#### Documentation

- `Documentation.pdf` : A description of the group's implementation of all three classifiers.

---

## Running the Application

- Only the `App.java` class is runnable.
- The `App.java` class defines methods (`run1()`, `run1()` and `run3()` ) that evaluate and run each of the classifiers on the provided training and testing data.
- In the `App.java` main method, by default, these methods are all called.
- The results of evaluating the classifiers as well as general status messages are outputted to the console.
- The results of classifying the provided testing data with the classifiers are written to text files (`run1.txt`, `run2.txt` and `run3.txt`) as a list of pairs of `<image name> <classification>`.
- Calls to each of these methods can be removed/commetted out in order to evaluate and run individual classifiers.

---
## Usage

### Performing Image Classification

- All three of the implemented classifiers extend the `MyClassifier.java` class, which defines the basic structure and methods of all of the three classifiers.
- One of these methods is `makeGuesses()`:

```java
public ArrayList<Tuple<String, String>> makeGuesses(VFSListDataset<FImage> dataset) { ... }
```

- After a classifier instance has been instantiated with training data, the `makeGuesses()` method can be used to annotate a set of unlabelled images.
- The images are passed in as a `VFSListDataset`, and the method returns an `ArrayList` of `Tuple`s, where each tuple is a `String, String` pair, with the first element being the name of the image, and the second being the annotation of this image.

---

