
var DTRegression = require('../index').DecisionTreeRegression;

describe('Decision tree regression', function () {
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

    test('Decision Tree classifier with sin function', function () {
        for (i = 0; i < x.length; ++i) {
            expect(estimations[i]).toBeCloseTo(y[i], 0);
        }
    });

    test('Export and import for decision tree classifier', function () {
        var model = JSON.parse(JSON.stringify(reg));

        var newClassifier = DTRegression.load(model);
        var newEstimations = newClassifier.predict(x);

        for (var i = 0; i < estimations.length; ++i) {
            expect(newEstimations[i]).toBe(estimations[i]);
        }
    });
});

