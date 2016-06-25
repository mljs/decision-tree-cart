"use strict";

var Utils = require("./Utils");
var Tree = require("./TreeNode");
var Matrix = require("ml-matrix");

class DecisionTreeClassifier {
    constructor() {
    }

    train(trainingSet, trainingLabels, options) {
        this.root = new Tree();
        if(!Matrix.isMatrix(trainingSet)) trainingSet = new Matrix(trainingSet);
        this.root.train(trainingSet, trainingLabels);
    }

    predict(toPredict) {
        var predictions = new Array(toPredict.length);

        for(var i = 0; i < toPredict.length; ++i) {
            predictions[i] = this.root.classify(toPredict[i]).maxRowIndex(0)[1];
        }

        return predictions;
    }

    score(X, y) {

    }
}



module.exports = DecisionTreeClassifier;