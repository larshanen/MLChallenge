# MLChallenge
## Assignment for Machine Learning - DSS Master

In a group of 4 students, with this assignment we will assess how well we can predict the year of publication of scientific papers, based on their metadata.

## Repo structure
### Data

Two JSON files will be provided by the university, containing a [train set](data\train.json) and a [test set](data\test.json). Evaluation of our predictions on the unlabeled test set, will be done through applying this code:

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
