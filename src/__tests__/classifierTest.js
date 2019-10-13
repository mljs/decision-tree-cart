import irisDataset from 'ml-dataset-iris';
import Matrix, { MatrixTransposeView } from 'ml-matrix';

import { DecisionTreeClassifier as DTClassifier } from '..';

var trainingSet = irisDataset.getNumbers();
var predictions = irisDataset
  .getClasses()
  .map((elem) => irisDataset.getDistinctClasses().indexOf(elem));

var options = {
  gainFunction: 'gini',
  maxDepth: 10,
  minNumSamples: 3
};

var classifier = new DTClassifier(options);
classifier.train(trainingSet, predictions);
var result = classifier.predict(trainingSet);

describe('Decision Tree Classifier', () => {
  it('Decision Tree classifier with iris dataset', () => {
    var correct = 0;
    for (var i = 0; i < result.length; ++i) {
      if (result[i] === predictions[i]) correct++;
    }

    var score = correct / result.length;
    expect(score).toBeGreaterThanOrEqual(0.7);
  });

  it('Export and import for decision tree classifier', () => {
    var model = JSON.parse(JSON.stringify(classifier));

    var newClassifier = DTClassifier.load(model);
    var newResult = newClassifier.predict(trainingSet);

    expect(newResult).toStrictEqual(result);
  });

  it('Check matrix transpose view', () => {
    var x = Matrix.checkMatrix(trainingSet).transpose();
    x = new MatrixTransposeView(x);

    var output = classifier.predict(x);

    var correct = 0;
    for (var i = 0; i < output.length; ++i) {
      if (output[i] === predictions[i]) correct++;
    }

    var score = correct / output.length;
    expect(score).toBeGreaterThanOrEqual(0.7);
  });
});
