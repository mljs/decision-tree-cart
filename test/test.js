"use strict";

var DTClassifier = require("..");

describe('basic functionality', function () {
    it('Decision Tree classifier', function () {
        var cases = [[6,148,72,35,0,33.6,0.627,5],
            [1.50,85,66.5,29,0,26.6,0.351,31],
            [8,183,64,0,0,23.3,0.672,32],
            [0.5,89,65.5,23,94,28.1,0.167,21],
            [0,137,40,35,168,43.1,2.288,33]];
        var predictions = [1, 0, 1, 0, 1];
        var classifier = new DTClassifier();
        classifier.train(cases, predictions);
        var result = classifier.predict(cases);

        console.log(result);

        (result[0]).should.be.equal(1);
        (result[1]).should.be.equal(1);
        (result[2]).should.be.equal(1);
        (result[3]).should.be.equal(0);
        (result[4]).should.be.equal(0);
    });
});