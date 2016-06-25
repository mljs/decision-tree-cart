"use strict";

var DTClassifier = require("..");
var irisDataset = require("ml-dataset-iris");

describe("basic functionality", function () {

    it("Decision Tree classifier with iris dataset", function () {
        var trainingSet = irisDataset.getNumbers();
        var predictions = irisDataset.getClasses().map(elem => irisDataset.getDistinctClasses().indexOf(elem));

        var classifier = new DTClassifier();
        classifier.train(trainingSet, predictions);
        var result = classifier.predict(trainingSet);

        var correct = 0;
        for(var i = 0 ; i < result.length; ++i) {
            if(result[i] == predictions[i]) correct++;
        }

        var score = correct / result.length;
        score.should.be.aboveOrEqual(0.7);
    });
});

describe("Utils", function () {
    it("Gini gain", function () {
    });
});