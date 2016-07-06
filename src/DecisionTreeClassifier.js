"use strict";

var Utils = require("./Utils");
var Tree = require("./TreeNode");
var Matrix = require("ml-matrix");

var gainFunctions = {
    gini: Utils.giniGain
};

var splitFunctions = {
    mean: Utils.mean
};

class DecisionTreeClassifier {

    /**
     * Create new Decision Tree Classifier with CART implementation with the given options
     * @param {Object} options
     * @param {String} [options.gainFunction="gini"] - gain function to get the best split, "gini" the only one supported.
     * @param {String} [options.splitFunction] - given two integers from a split feature, get the value to split, "mean" the only one supported.
     * @param {Number} [options.minNumSamples] - minimum number of samples to create a leaf node to decide a class. Default 3.
     * @param {Number} [options.maxDepth] - Max depth of the tree. Default Infinity.
     */
    constructor(options) {
        if(options === undefined) options = {};
        options.gainFunction = gainFunctions[options.gainFunction];
        if(options.gainFunction === undefined) options.gainFunction = gainFunctions["gini"];
        if(options.splitFunction === undefined) options.splitFunction = splitFunctions["mean"];
        if(options.minNumSamples === undefined) options.minNumSamples = 3;
        if(options.maxDepth === undefined) options.maxDepth = Infinity;

        options.kind = "classifier";
        this.options = options;
    }

    /**
     * Train the decision tree with the given training set and labels.
     * @param {Matrix} trainingSet
     * @param {Array} trainingLabels
     */
    train(trainingSet, trainingLabels) {
        this.root = new Tree(this.options);
        if(!Matrix.isMatrix(trainingSet)) trainingSet = new Matrix(trainingSet);
        this.root.train(trainingSet, trainingLabels, 0, null);
    }

    /**
     * Predicts the output given the matrix to predict.
     * @param {Matrix} toPredict 
     * @returns {Array} predictions
     */
    predict(toPredict) {
        var predictions = new Array(toPredict.length);

        for(var i = 0; i < toPredict.length; ++i) {
            predictions[i] = this.root.classify(toPredict[i]).maxRowIndex(0)[1];
        }

        return predictions;
    }
}

module.exports = DecisionTreeClassifier;