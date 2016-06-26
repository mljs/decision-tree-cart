"use strict";

var DTClassifier = require("..");
var irisDataset = require("ml-dataset-iris");
var Utils = require("../src/Utils");

describe("Basic functionality", function () {

    it("Decision Tree classifier with iris dataset", function () {
        var trainingSet = irisDataset.getNumbers();
        var predictions = irisDataset.getClasses().map(elem => irisDataset.getDistinctClasses().indexOf(elem));

        var classifier = new DTClassifier({
            gainFunction: Utils.giniGain,
            minNumSamples: 40,
            maxDepth: Infinity
        });
        classifier.train(trainingSet, predictions);
        var result = classifier.predict(trainingSet);

        var correct = 0;
        for(var i = 0 ; i < result.length; ++i) {
            if(result[i] == predictions[i]) correct++;
        }

        var score = correct / result.length;
        console.log(score);
        score.should.be.aboveOrEqual(0.7);
    });
});

describe("Utils", function () {
    it("Gini gain", function () {
        Utils.giniGain([0, 1, 0, 1, 0, 1], {
            greater: [0, 0],
            lesser: [1, 1, 1, 0]
        }).should.be.approximately(0.25, 0.001);
    });

    it("Get number of classes", function () {
        Utils.getNumberOfClasses([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).should.be.equal(10);
    });
});