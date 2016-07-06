"use strict";

var Utils = require("./Utils");
var Tree = require("./TreeNode");
var Matrix = require("ml-matrix");

var costFunctions = {
    regression: Utils.regressionError
};

var splitFunctions = {
    mean: Utils.mean
};


class DecisionTreeRegression {

    /**
     * Create new Decision Tree Regression with CART implementation with the given options.
     * @param {Object} options
     * @param {String} [options.gainFunction="regression"] - gain function to get the best split, "regression" the only one supported.
     * @param {String} [options.splitFunction] - given two integers from a split feature, get the value to split, "mean" the only one supported.
     * @param {Number} [options.minNumSamples] - minimum number of samples to create a leaf node to decide a class. Default 3.
     * @param {Number} [options.maxDepth] - Max depth of the tree. Default Infinity.
     */
    constructor(options) {
        if(options === undefined) options = {};
        options.gainFunction = costFunctions[options.gainFunction];
        if(options.gainFunction === undefined) options.gainFunction = costFunctions["regression"];
        if(options.splitFunction === undefined) options.splitFunction = splitFunctions["mean"];
        if(options.minNumSamples === undefined) options.minNumSamples = 3;
        if(options.maxDepth === undefined) options.maxDepth = Infinity;

        options.kind = "regression";
        this.options = options;
    }

    /**
     * Train the decision tree with the given training set and values.
     * @param {Matrix} trainingSet
     * @param {Array} trainingValues
     */
    train(trainingSet, trainingValues) {
        this.root = new Tree(this.options);
        if(trainingSet[0].length === undefined) trainingSet = Matrix.columnVector(trainingSet);
        if(!Matrix.isMatrix(trainingSet)) trainingSet = new Matrix(trainingSet);
        this.root.train(trainingSet, trainingValues, 0);
    }

    /**
     * Predicts the values given the matrix to predict.
     * @param {Matrix} toPredict
     * @returns {Array} predictions
     */
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