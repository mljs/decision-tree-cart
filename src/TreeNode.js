"use strict";

var Matrix = require("ml-matrix");
var Utils = require("./Utils");

class TreeNode {
    constructor(options) {
        /*if(options === undefined) options = {};
        if(options.gainFunction === undefined) options.gainFunction = Utils.giniGain;
        if(options.splitFunction === undefined) options.splitFunction = mean;
        if(options.minNumSamples === undefined) options.minNumSamples = 3;
        if(options.maxDepth === undefined) options.maxDepth = Infinity;*/

        this.left = undefined;
        this.right = undefined;
        this.distribution = undefined;
        this.splitValue = undefined;
        this.splitColumn = undefined;
        this.gain = undefined;
        this.options = options;
    }

    bestSplit(XTranspose, y) {
        var maxGain = -Infinity;
        var maxColumn = undefined;
        var maxValue = undefined;

        for(var i = 0; i < XTranspose.rows; ++i) {
            var currentFeature = XTranspose[i];
            var splitValues = this.featureSplit(currentFeature, y);
            for(var j = 0 ; j < splitValues.length; ++j) {
                var currentSplitVal = splitValues[j];
                var splitted = this.split(currentFeature, y, currentSplitVal);
                
                var gain = this.options.gainFunction(y, splitted);
                if(gain > maxGain) {
                    maxColumn = i;
                    maxValue = currentSplitVal;
                    maxGain = gain;
                }
            }
        }

        return {
            maxGain: maxGain,
            maxColumn: maxColumn,
            maxValue: maxValue
        };
    }
    
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
    
    calculatePrediction(y) {
        this.distribution = Utils.toDiscreteDistribution(y, Utils.getNumberOfClasses(y));
        if(this.distribution.columns == 0) {
            throw new TypeError("Error on calculate the prediction");
        }
    }

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
            (this.gain > 0.01 || this.gain != parentGain) &&
            (splittedMatrix["lesserX"].length > 0 && splittedMatrix["greaterX"].length > 0)) {
            this.left = new TreeNode(this.options);
            this.right = new TreeNode(this.options);

            var lesserX = new Matrix(splittedMatrix["lesserX"]);
            var greaterX = new Matrix(splittedMatrix["greaterX"]);

            this.left.train(lesserX, splittedMatrix["lesserY"], currentDepth + 1, this.gain);
            this.right.train(greaterX, splittedMatrix["greaterY"], currentDepth + 1,this.gain);
        } else {
            this.calculatePrediction(y);
        }
    }

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