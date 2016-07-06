"use strict";

var Matrix = require("ml-matrix");
var Utils = require("./Utils");

class TreeNode {

    /**
     * Constructor for a tree node given the options received on the main classes (DecisionTreeClassifier, DecisionTreeRegression)
     * @param options
     */
    constructor(options) {

        this.left = undefined;
        this.right = undefined;
        this.distribution = undefined;
        this.splitValue = undefined;
        this.splitColumn = undefined;
        this.gain = undefined;
        this.options = options;
    }
    
    /**
     * Function that retrieve the best feature to make the split.
     * @param XTranspose - Training set transposed
     * @param y - labels or values (depending of the decision tree)
     * @returns {{maxGain: {Number}, maxColumn: {Number}, maxValue: {Number}}} -
     *           return tree values, the best gain, column and the split value.
     */
    bestSplit(XTranspose, y) {

        /**
         * Depending in the node tree class, we set the variables to check information gain (to classify)
         * or error (for regression)
         */
        var bestGain = this.options.kind === "classifier" ? -Infinity : Infinity;
        var check = this.options.kind === "classifier" ? (a, b) => a > b : (a, b) => a < b;


        var maxColumn = undefined;
        var maxValue = undefined;

        for(var i = 0; i < XTranspose.rows; ++i) {
            var currentFeature = XTranspose[i];
            var splitValues = this.featureSplit(currentFeature, y);
            for(var j = 0 ; j < splitValues.length; ++j) {
                var currentSplitVal = splitValues[j];
                var splitted = this.split(currentFeature, y, currentSplitVal);
                
                var gain = this.options.gainFunction(y, splitted);
                if(check(gain, bestGain)) {
                    maxColumn = i;
                    maxValue = currentSplitVal;
                    bestGain = gain;
                }
            }
        }

        return {
            maxGain: bestGain,
            maxColumn: maxColumn,
            maxValue: maxValue
        };
    }

    /**
     * Makes the split of the training labels or values from the training set feature given a split value.
     * @param {Array} x - Training set feature
     * @param {Array} y - Training set value or label
     * @param {Number} splitValue
     * @returns {{greater: Array, lesser: Array}}
     */

    split(x, y, splitValue) {
        var lesser = [];
        var greater = [];
        
        for(var i = 0; i < x.length; ++i) {
            if(x[i] < splitValue) {
                lesser.push(y[i]);
            } else {
                greater.push(y[i]);
            }
        }
        
        return {
            greater: greater,
            lesser: lesser
        };
    }

    /**
     * Calculates the possible points to split over the tree given a training set feature and corresponding labels or values.
     * @param {Array} x - Training set feature
     * @param {Array} y - Training set value or label
     * @returns {Array} possible split values.
     */
    featureSplit(x, y) {
        var splitValues = [];
        var arr = Utils.zip(x, y);
        arr.sort(function (a, b) {
            return a[0] - b[0];
        });

        for(var i = 1; i < arr.length; ++i) {
            if(arr[i - 1][1] != arr[i][1]) {
                splitValues.push(this.options.splitFunction(arr[i - 1][0], arr[i][0]));
            }
        }

        return splitValues;
    }

    /**
     * Calculate the predictions of a leaf tree node given the training labels or values
     * @param {Array} y
     */
    calculatePrediction(y) {
        if(this.options.kind === "classifier") {
            this.distribution = Utils.toDiscreteDistribution(y, Utils.getNumberOfClasses(y));
            if (this.distribution.columns == 0) {
                throw new TypeError("Error on calculate the prediction");
            }
        } else {
            this.distribution = y.reduce((a, b) => a + b, 0) / y.length;
        }
    }

    /**
     * Train a node given the training set and labels, because it trains recursively, it also receive
     * the current depth of the node, parent gain to avoid infinite recursion and boolean value to check if
     * the training set is transposed.
     * @param {Matrix} X - Training set (could be transposed or not given transposed).
     * @param {Array} y - Training labels or values.
     * @param {Number} currentDepth - Current depth of the node.
     * @param {Number} parentGain - parent node gain or error.
     */
    train(X, y, currentDepth, parentGain) {
        if(X.rows <= this.options.minNumSamples) {
            this.calculatePrediction(y);
            return;
        }
        if(parentGain == undefined) parentGain = 0.0;

        var XTranspose = X.transpose();
        var split = this.bestSplit(XTranspose, y);

        this.splitValue = split["maxValue"];
        this.splitColumn = split["maxColumn"];
        this.gain = split["maxGain"];

        var splittedMatrix = Utils.matrixSplitter(X, y, this.splitColumn, this.splitValue);

        if(currentDepth < this.options.maxDepth &&
            (this.gain > 0.01 && this.gain != parentGain) &&
            (splittedMatrix["lesserX"].length > 0 && splittedMatrix["greaterX"].length > 0)) {
            this.left = new TreeNode(this.options);
            this.right = new TreeNode(this.options);

            var lesserX = new Matrix(splittedMatrix["lesserX"]);
            var greaterX = new Matrix(splittedMatrix["greaterX"]);

            this.left.train(lesserX, splittedMatrix["lesserY"], currentDepth + 1, this.gain);
            this.right.train(greaterX, splittedMatrix["greaterY"], currentDepth + 1, this.gain);
        } else {
            this.calculatePrediction(y);
        }
    }

    /**
     * Calculates the prediction of a given element.
     * @param {Array} row
     * @returns {Number|Array} prediction 
     *          * if a node is a classifier returns an array of probabilities of each class.
     *          * if a node is for regression returns a number with the prediction.
     */
    classify(row) {
        if(this.right && this.left) {
            if(row[this.splitColumn] < this.splitValue) {
                return this.left.classify(row);
            } else {
                return this.right.classify(row);
            }
        }

        return this.distribution;
    }
}

module.exports = TreeNode;