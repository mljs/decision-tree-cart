import { DecisionTreeClassifier as DTClassifier } from '..';

var classifier = new DTClassifier({
  gainFunction: 'gini',
  maxDepth: 10,
  minNumSamples: 1
});

// Returns index of marked row

var trainingSet = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1]
]

var predictions = [0, 1, 2, 3]

classifier.train(trainingSet, predictions);

var result = classifier.predict(trainingSet);

describe('Decision Tree Classifier', () => {
  test('Decision Tree classifier with simple dataset', () => {
    var correct = 0;
    for (var i = 0; i < result.length; ++i) {
      if (result[i] === predictions[i]) correct++;
    }

    var score = correct / result.length;
    expect(score).toBe(1.0)
  });

  test('Export and import for decision simple classifier', () => {
    var model = JSON.parse(JSON.stringify(classifier));

    var newClassifier = DTClassifier.load(model);
    var newResult = newClassifier.predict(trainingSet);

    expect(newResult).toEqual(result);
  });
});
