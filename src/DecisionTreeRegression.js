"use strict";

var Utils = require("./Utils");
var Tree = require("./TreeNode");
var Matrix = require("ml-matrix");

var costFunctions = {
    regression: Utils.regressionError
};

class DecisionTreeRegression {
    constructor(options) {
        if(options === undefined) options = {};
        options.gainFunction = costFunctions[options.gainFunction];
        if(options.gainFunction === undefined) options.gainFunction = costFunctions["regression"];
        if(options.splitFunction === undefined) options.splitFunction = Utils.mean;
        if(options.minNumSamples === undefined) options.minNumSamples = 3;
        if(options.maxDepth === undefined) options.maxDepth = Infinity;

        options.kind = "regression";
        this.options = options;
    }

    train(trainingSet, trainingValues) {
        this.root = new Tree(this.options);
        if(trainingSet[0].length === undefined) trainingSet = Matrix.columnVector(trainingSet);
        if(!Matrix.isMatrix(trainingSet)) trainingSet = new Matrix(trainingSet);
        this.root.train(trainingSet, trainingValues, 0);
    }

    predict(toPredict) {
        if(toPredict[0].length === undefined) toPredict = Matrix.columnVector(toPredict);
        var predictions = new Array(toPredict.length);

        for(var i = 0; i < toPredict.length; ++i) {
            predictions[i] = this.root.classify(toPredict[i]);
        }

        return predictions;
    }
}

module.exports = DecisionTreeRegression;