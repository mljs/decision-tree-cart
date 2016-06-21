"use strict";

var Matrix = require("ml-matrix");

class TreeNode {
    constructor() {
        this.left = undefined;
        this.right = undefined;
        this.prediction = undefined;
        this.splitFunction = mean;
        this.splitValue = undefined;
        this.splitColumn = undefined;
        this.gain = undefined;
        this.gainFunction = undefined;
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
                
                var gain = this.gainFunction(y, splitted);
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
        var arr = zip(x, y);
        arr.sort(function (a, b) {
            return a[0] - b[0];
        });

        for(var i = 1; i < arr.length; ++i) {
            if(arr[i - 1][1] != arr[i][1]) {
                splitValues.push(this.splitFunction(arr[i - 1][0], arr[i][0]));
            }
        }

        return splitValues;
    }

    train(X, y) {
        this.train(X, y, 0.0);
    }

    train(X, y, parentGain) {
        var XTranspose = X.transpose();
        var split = this.bestSplit(XTranspose, y);

        this.splitValue = split["maxValue"];
        this.splitColumn = split["maxColumn"];
        this.gain = split["gain"];


    }

    classify(row) {
        if(this.right && this.left) {
            if(row[this.splitColumn] < this.splitValue) {
                return this.left.classify(row);
            } else {
                return this.right.classify(row);
            }
        }

        return this.prediction;
    }
}

function matrixSplitter(X, y, column, value) {
    var lesserX = [];
    var greaterX = [];
    var lesserY = [];
    var greaterY = [];

    for(var i = 0; i < X.rows; ++i) {
        if(X[i][column] < value) {
            lesserX.push(X[i]);
            lesserY.push(y[i]);
        } else {
            greaterX.push(X[i]);
            greaterY.push(y[i]);
        }
    }

    return {
        greaterX: greaterX,
        greaterY: greaterY,
        lesserX: lesserX,
        lesserY: lesserY
    };
}

function mean(a, b) {
    return (a + b) / 2;
}

function zip(a, b) {
    if(a.length !== b.length) {
        throw new TypeError("Error on zip: the size of a: " + a.length + " is different from b: " + b.length);
    }

    var ret = new Array(a.length);
    for(var i = 0; i < a.length; ++i) {
        ret[i] = [a[i], b[i]];
    }

    return ret;
}
