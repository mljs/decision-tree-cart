"use strict";

var Matrix = require("ml-matrix");

function toDiscreteDistribution(array, nClasses) {
    var counts = new Array(nClasses).fill(0);
    for(var i = 0; i < array.length; ++i) {
        counts[array[i]] += 1 / array.length;
    }

    return Matrix.rowVector(counts);
}

function giniImpurity(array) {
    if(array.length == 0) {
        return 0;
    }

    var probabilities = toDiscreteDistribution(array, getNumberOfClasses(array))[0];

    var sum = 0.0;
    for(var i = 0; i < probabilities.length; ++i) {
        sum += probabilities[i] * probabilities[i];
    }

    return 1 - sum;
}

function getNumberOfClasses(array) {
    return array.filter(function(val, i, arr) {
        return arr.indexOf(val) === i;
    }).length;
}

function giniGain(array, splitted) {
    var splitsImpurity = 0.0;
    var splits = ["greater", "lesser"];

    for(var i = 0; i < splits.length; ++i) {
        var currentSplit = splitted[splits[i]];
        splitsImpurity += giniImpurity(currentSplit) * currentSplit.length / array.length;
    }

    return giniImpurity(array) - splitsImpurity;
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


module.exports = {
    toDiscreteDistribution: toDiscreteDistribution,
    getNumberOfClasses: getNumberOfClasses,
    giniGain: giniGain,
    zip: zip,
    mean: mean,
    matrixSplitter: matrixSplitter
};