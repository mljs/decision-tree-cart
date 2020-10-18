import { DecisionTreeClassifier as DTClassifier } from '..';

let classifier = new DTClassifier({
  gainFunction: 'gini',
  maxDepth: 10,
  minNumSamples: 1,
});

// Returns index of marked row

let trainingSet = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
];

let predictions = [0, 1, 2, 3];

classifier.train(trainingSet, predictions);

let result = classifier.predict(trainingSet);

describe('Decision Tree Classifier', () => {
  it('Decision Tree classifier with simple dataset', () => {
    const correct = result.reduce((prev, value, index) => {
      return value === predictions[index] ? prev + 1 : prev;
    }, 0);
    let score = correct / result.length;
    expect(score).toBe(1.0);
  });

  it('Export and import for decision simple classifier', () => {
    let model = JSON.parse(JSON.stringify(classifier));

    let newClassifier = DTClassifier.load(model);
    let newResult = newClassifier.predict(trainingSet);

    expect(newResult).toStrictEqual(result);
  });
});
