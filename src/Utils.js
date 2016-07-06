'use strict';

var Matrix = require('ml-matrix');

/**
 * return an array of probabilities of each class
 * @param {Array} array - contains the classes
 * @param {Number} numberOfClasses
 * @returns {Matrix} - rowVector of probabilities.
 */
function toDiscreteDistribution(array, numberOfClasses) {
    var counts = new Array(numberOfClasses).fill(0);
    for (var i = 0; i < array.length; ++i) {
        counts[array[i]] += 1 / array.length;
    }

    return Matrix.rowVector(counts);
}

/**
 * Retrieves the impurity of array of predictions
 * @param {Array} array - predictions.
 * @returns {number} Gini impurity
 */
function giniImpurity(array) {
    if (array.length === 0) {
        return 0;
    }

    var probabilities = toDiscreteDistribution(array, getNumberOfClasses(array))[0];

    var sum = 0.0;
    for (var i = 0; i < probabilities.length; ++i) {
        sum += probabilities[i] * probabilities[i];
    }

    return 1 - sum;
}

/**
 * Return the number of classes given the array of predictions.
 * @param {Array} array - predictions.
 * @returns {Number} Number of classes.
 */
function getNumberOfClasses(array) {
    return array.filter(function (val, i, arr) {
        return arr.indexOf(val) === i;
    }).length;
}

/**
 * Calculates the Gini Gain of an array of predictions and those predictions splitted by a feature.
 * @para {Array} array - Predictions
 * @param {Object} splitted - Object with elements "greater" and "lesser" that contains an array of predictions splitted.
 * @returns {number} - Gini Gain.
 */

function giniGain(array, splitted) {
    var splitsImpurity = 0.0;
    var splits = ['greater', 'lesser'];

    for (var i = 0; i < splits.length; ++i) {
        var currentSplit = splitted[splits[i]];
        splitsImpurity += giniImpurity(currentSplit) * currentSplit.length / array.length;
    }

    return giniImpurity(array) - splitsImpurity;
}

/**
 * Calculates the squared error of a predictions values.
 * @param {Array} array - predictions values
 * @returns {Number} squared error.
 */
function squaredError(array) {
    var mean = array.reduce((a, b) => a + b, 0) / array.length;
    return array.map(elem => (elem - mean) * (elem - mean)).reduce((a, b) => a + b, 0);
}

/**
 * Calculates the sum of squared error of the two arrays that contains the splitted values.
 * @param {Array} array - this argument is no necessary but is to fit with the main interface.
 * @param {Object} splitted - Object with elements "greater" and "lesser" that contains an array of predictions splitted.
 * @returns {number} - sum of squared errors.
 */
function regressionError(array, splitted) {
    var error = 0.0;
    var splits = ['greater', 'lesser'];

    for (var i = 0; i < splits.length; ++i) {
        var currentSplit = splitted[splits[i]];
        error += squaredError(currentSplit);
    }
    return error;
}

/**
 * Split the training set and values from a given column of the training set if is less than a value
 * @param {Matrix} X - Training set.
 * @param {Array} y - Training values.
 * @param {Number} column - Column to split.
 * @param {Number} value - value to split the Training set and values.
 * @returns {{greaterX: Array, greaterY: Array, lesserX: Array, lesserY: Array}} - Object that contains the splitted values.
 */
function matrixSplitter(X, y, column, value) {
    var lesserX = [];
    var greaterX = [];
    var lesserY = [];
    var greaterY = [];

    for (var i = 0; i < X.rows; ++i) {
        if (X[i][column] < value) {
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

/**
 * Calculates the mean between two values
 * @param {Number} a
 * @param {Number} b
 * @returns {number}
 */
function mean(a, b) {
    return (a + b) / 2;
}

/**
 * Returns a list of tuples that contains the i-th element of each array.
 * @param {Array} a
 * @param {Array} b
 * @returns {Array} - list of tuples.
 */
function zip(a, b) {
    if (a.length !== b.length) {
        throw new TypeError('Error on zip: the size of a: ' + a.length + ' is different from b: ' + b.length);
    }

    var ret = new Array(a.length);
    for (var i = 0; i < a.length; ++i) {
        ret[i] = [a[i], b[i]];
    }

    return ret;
}

module.exports = {
    toDiscreteDistribution: toDiscreteDistribution,
    getNumberOfClasses: getNumberOfClasses,
    giniGain: giniGain,
    regressionError: regressionError,
    zip: zip,
    mean: mean,
    matrixSplitter: matrixSplitter
};
