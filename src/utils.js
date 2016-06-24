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

function giniGain(array, splited) {
    var splitsImpurity = 0.0;
    var splits = ["greater", "lesser"];

    for(var i = 0; i < splits.length; ++i) {
        var currentSplit = splited[splits[i]];
        splitsImpurity += giniImpurity(currentSplit) * currentSplit.length / array.length;
    }

    return giniImpurity(array) - splitsImpurity;
}

module.exports = {
    toDiscreteDistribution: toDiscreteDistribution,
    getNumberOfClasses: getNumberOfClasses,
    giniGain: giniGain
};