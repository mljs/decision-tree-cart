import Matrix from 'ml-matrix';
import meanArray from 'ml-array-mean';

/**
 * @private
 * return an array of probabilities of each class
 * @param {Array} array - contains the classes
 * @param {number} numberOfClasses
 * @return {Matrix} - rowVector of probabilities.
 */
export function toDiscreteDistribution(array, numberOfClasses) {
    var counts = new Array(numberOfClasses).fill(0);
    for (var i = 0; i < array.length; ++i) {
        counts[array[i]] += 1 / array.length;
    }

    return Matrix.rowVector(counts);
}

/**
 * @private
 * Retrieves the impurity of array of predictions
 * @param {Array} array - predictions.
 * @return {number} Gini impurity
 */
export function giniImpurity(array) {
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
 * @private
 * Return the number of classes given the array of predictions.
 * @param {Array} array - predictions.
 * @return {number} Number of classes.
 */
export function getNumberOfClasses(array) {
    return array.filter(function (val, i, arr) {
        return arr.indexOf(val) === i;
    }).length;
}

/**
 * @private
 * Calculates the Gini Gain of an array of predictions and those predictions splitted by a feature.
 * @para {Array} array - Predictions
 * @param {object} splitted - Object with elements "greater" and "lesser" that contains an array of predictions splitted.
 * @return {number} - Gini Gain.
 */

export function giniGain(array, splitted) {
    var splitsImpurity = 0.0;
    var splits = ['greater', 'lesser'];

    for (var i = 0; i < splits.length; ++i) {
        var currentSplit = splitted[splits[i]];
        splitsImpurity += giniImpurity(currentSplit) * currentSplit.length / array.length;
    }

    return giniImpurity(array) - splitsImpurity;
}

/**
 * @private
 * Calculates the squared error of a predictions values.
 * @param {Array} array - predictions values
 * @return {number} squared error.
 */
export function squaredError(array) {
    var l = array.length;

    var m = meanArray(array);
    var squaredError = 0.0;

    for (var i = 0; i < l; ++i) {
        var currentElement = array[i];
        squaredError += (currentElement - m) * (currentElement - m);
    }

    return squaredError;
}

/**
 * @private
 * Calculates the sum of squared error of the two arrays that contains the splitted values.
 * @param {Array} array - this argument is no necessary but is used to fit with the main interface.
 * @param {object} splitted - Object with elements "greater" and "lesser" that contains an array of predictions splitted.
 * @return {number} - sum of squared errors.
 */
export function regressionError(array, splitted) {
    var error = 0.0;
    var splits = ['greater', 'lesser'];

    for (var i = 0; i < splits.length; ++i) {
        var currentSplit = splitted[splits[i]];
        error += squaredError(currentSplit);
    }
    return error;
}

/**
 * @private
 * Split the training set and values from a given column of the training set if is less than a value
 * @param {Matrix} X - Training set.
 * @param {Array} y - Training values.
 * @param {number} column - Column to split.
 * @param {number} value - value to split the Training set and values.
 * @return {object} - Object that contains the splitted values.
 */
export function matrixSplitter(X, y, column, value) {
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
 * @private
 * Calculates the mean between two values
 * @param {number} a
 * @param {number} b
 * @return {number}
 */
export function mean(a, b) {
    return (a + b) / 2;
}

/**
 * @private
 * Returns a list of tuples that contains the i-th element of each array.
 * @param {Array} a
 * @param {Array} b
 * @return {Array} list of tuples.
 */
export function zip(a, b) {
    if (a.length !== b.length) {
        throw new TypeError('Error on zip: the size of a: ' + a.length + ' is different from b: ' + b.length);
    }

    var ret = new Array(a.length);
    for (var i = 0; i < a.length; ++i) {
        ret[i] = [a[i], b[i]];
    }

    return ret;
}
