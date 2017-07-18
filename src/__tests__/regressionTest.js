import {DecisionTreeRegression as DTRegression} from '..';

var x = new Array(100);
var y = new Array(100);
var val = 0.0;
for (var i = 0; i < x.length; ++i) {
    x[i] = val;
    y[i] = Math.sin(x[i]);
    val += 0.01;
}

var reg = new DTRegression();
reg.train(x, y);
var estimations = reg.predict(x);

describe('Decision tree regression', () => {
    test('Decision Tree classifier with sin function', () => {
        for (var i = 0; i < x.length; ++i) {
            expect(estimations[i]).toBeCloseTo(y[i], 0);
        }
    });

    test('Export and import for decision tree classifier', () => {
        var model = JSON.parse(JSON.stringify(reg));

        var newClassifier = DTRegression.load(model);
        var newEstimations = newClassifier.predict(x);

        expect(newEstimations).toEqual(estimations);
    });
});

