import irisDataset from 'ml-dataset-iris';
import {DecisionTreeClassifier as DTClassifier} from '..';

describe('Decision Tree Classifier', function () {
    var trainingSet = irisDataset.getNumbers();
    var predictions = irisDataset.getClasses().map(elem => irisDataset.getDistinctClasses().indexOf(elem));

    var options = {
        gainFunction: 'gini',
        maxDepth: 10,
        minNumSamples: 3
    };

    var classifier = new DTClassifier(options);
    classifier.train(trainingSet, predictions);
    var result = classifier.predict(trainingSet);

    test('Decision Tree classifier with iris dataset', function () {
        var correct = 0;
        for (var i = 0; i < result.length; ++i) {
            if (result[i] === predictions[i]) correct++;
        }

        var score = correct / result.length;
        expect(score).toBeGreaterThanOrEqual(0.7);
    });

    test('Export and import for decision tree classifier', function () {
        var model = JSON.parse(JSON.stringify(classifier));

        var newClassifier = DTClassifier.load(model);
        var newResult = newClassifier.predict(trainingSet);

        for (var i = 0; i < result.length; ++i) {
            expect(newResult[i]).toBe(result[i]);
        }
    });
});
