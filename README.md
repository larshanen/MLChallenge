# MLChallenge
## Assignment for Machine Learning - DSS Master

In a group of 4 students, with this assignment we will assess how well we can predict the year of publication of scientific papers, based on their metadata.

## Repo structure
### Data

Two JSON files will be provided by the university, containing a [train set](data/train.json) and a [test set](data/test.json). Evaluation of our predictions on the unlabeled test set, will be done through applying this code:

```
import json
from sklearn.metrics import mean_absolute_error
import numpy as np
import json

def evaluate(gold_path, pred_path):
    gold = np.array([ x['year'] for x in json.load(open(gold_path)) ]).astype(float)
    pred = np.array([ x['year'] for x in json.load(open(pred_path)) ]).astype(float)

    return mean_absolute_error(gold, pred)
```

### Method
Part (X%) of the provided train set will be used as a validation set. As the test set remains unlabeled until after the university deadline, this validation set is what we use to assess the quality of our predictions. The assessment is done by calculating the Mean Squared Error (MSE).

## Tasks
### Beforehand:
- [x] Set up Github environment
- [x] Document baseline code provided by uni

### Round 1 (22 nov): Exploratory Data Analysis
- [ ] Lars: Make overview of NLP methods relevant for assignment, setup a notebook for each task/team member in round 1 and apply first changes to baseline code
- [ ] Alysha: Define and describe metadata (entrytype, title, editor, year, publisher, author, abstract), think of type (e.g. categorical, numerical) and purpose of the 'column'
- [ ] Bell: Missing data analysis, if ready help Alysha if needed
- [ ] Rick: Missing data heatmap using Bell's analysis, start model comparisons

### Round 2 (...): ...
- [ ] ...
