import { getClasses, getDistinctClasses, getNumbers } from 'ml-dataset-iris';
import { Matrix, MatrixTransposeView } from 'ml-matrix';

import { DecisionTreeClassifier as DTClassifier } from '..';

let trainingSet = getNumbers();
let predictions = getClasses().map((elem) =>
  getDistinctClasses().indexOf(elem),
);

let options = {
  gainFunction: 'gini',
  maxDepth: 10,
  minNumSamples: 3,
};

let classifier = new DTClassifier(options);
classifier.train(trainingSet, predictions);
let result = classifier.predict(trainingSet);

describe('Decision Tree Classifier', () => {
  it('Decision Tree classifier with iris dataset', () => {
    const correct = result.reduce((prev, value, index) => {
      return value === predictions[index] ? prev + 1 : prev;
    }, 0);

    let score = correct / result.length;
    expect(score).toBeGreaterThanOrEqual(0.7);
  });

  it('Export and import for decision tree classifier', () => {
    let model = JSON.parse(JSON.stringify(classifier));

    let newClassifier = DTClassifier.load(model);
    let newResult = newClassifier.predict(trainingSet);

    expect(newResult).toStrictEqual(result);
  });

  it('Check matrix transpose view', () => {
    let x = Matrix.checkMatrix(trainingSet).transpose();
    x = new MatrixTransposeView(x);

    let output = classifier.predict(x);

    const correct = output.reduce((prev, value, index) => {
      return value === predictions[index] ? prev + 1 : prev;
    }, 0);

    let score = correct / output.length;
    expect(score).toBeGreaterThanOrEqual(0.7);
  });
});
