import mean from 'ml-array-mean';
import { Matrix } from 'ml-matrix';

import * as Utils from './utils';

const gainFunctions = {
  gini: Utils.giniGain,
  regression: Utils.regressionError,
};

const splitFunctions = {
  mean: Utils.mean,
};

export default class TreeNode {
  /**
   * @private
   * Constructor for a tree node given the options received on the main classes (DecisionTreeClassifier, DecisionTreeRegression)
   * @param {object|TreeNode} options for loading
   * @constructor
   */
  constructor(options) {
    // options parameters
    this.kind = options.kind;
    this.gainFunction = options.gainFunction;
    this.splitFunction = options.splitFunction;
    this.minNumSamples = options.minNumSamples;
    this.maxDepth = options.maxDepth;
  this.gainThreshold = options.gainThreshold || 0;
  }

  /**
   * @private
   * Function that retrieve the best feature to make the split.
   * @param {Matrix} XTranspose - Training set transposed
   * @param {Array} y - labels or values (depending of the decision tree)
   * @return {object} - return tree values, the best gain, column and the split value.
   */
  bestSplit(XTranspose, y) {
    // Depending in the node tree class, we set the variables to check information gain (to classify)
    // or error (for regression)

    let bestGain = this.kind === 'classifier' ? -Infinity : Infinity;
    let check = this.kind === 'classifier' ? (a, b) => a > b : (a, b) => a < b;

    let maxColumn;
    let maxValue;
    let numberSamples;

    for (let i = 0; i < XTranspose.rows; ++i) {
      let currentFeature = XTranspose.getRow(i);
      let splitValues = this.featureSplit(currentFeature, y);
      for (let j = 0; j < splitValues.length; ++j) {
        let currentSplitVal = splitValues[j];
        let splitted = this.split(currentFeature, y, currentSplitVal);

        let gain = gainFunctions[this.gainFunction](y, splitted);
        if (check(gain, bestGain)) {
          maxColumn = i;
          maxValue = currentSplitVal;
          bestGain = gain;
          numberSamples = currentFeature.length;
        }
      }
    }

    return {
      maxGain: bestGain,
      maxColumn: maxColumn,
      maxValue: maxValue,
      numberSamples: numberSamples,
    };
  }

  /**
   * @private
   * Makes the split of the training labels or values from the training set feature given a split value.
   * @param {Array} x - Training set feature
   * @param {Array} y - Training set value or label
   * @param {number} splitValue
   * @return {object}
   */
  split(x, y, splitValue) {
    let lesser = [];
    let greater = [];

    for (let i = 0; i < x.length; ++i) {
      if (x[i] < splitValue) {
        lesser.push(y[i]);
      } else {
        greater.push(y[i]);
      }
    }

    return {
      greater: greater,
      lesser: lesser,
    };
  }

  /**
   * @private
   * Calculates the possible points to split over the tree given a training set feature and corresponding labels or values.
   * @param {Array} x - Training set feature
   * @param {Array} y - Training set value or label
   * @return {Array} possible split values.
   */
  featureSplit(x, y) {
    let splitValues = [];
    let arr = Utils.zip(x, y);
    arr.sort((a, b) => {
      return a[0] - b[0];
    });

    for (let i = 1; i < arr.length; ++i) {
      if (arr[i - 1][1] !== arr[i][1]) {
        splitValues.push(
          splitFunctions[this.splitFunction](arr[i - 1][0], arr[i][0]),
        );
      }
    }

    return splitValues;
  }

  /**
   * @private
   * Calculate the predictions of a leaf tree node given the training labels or values
   * @param {Array} y
   */
  calculatePrediction(y) {
    if (this.kind === 'classifier') {
      this.distribution = Utils.toDiscreteDistribution(
        y,
        Utils.getNumberOfClasses(y),
      );
      if (this.distribution.columns === 0) {
        throw new TypeError('Error on calculate the prediction');
      }
    } else {
      this.distribution = mean(y);
    }
  }

  /**
   * @private
   * Train a node given the training set and labels, because it trains recursively, it also receive
   * the current depth of the node, parent gain to avoid infinite recursion and boolean value to check if
   * the training set is transposed.
   * @param {Matrix} X - Training set (could be transposed or not given transposed).
   * @param {Array} y - Training labels or values.
   * @param {number} currentDepth - Current depth of the node.
   * @param {number} parentGain - parent node gain or error.
   */
  train(X, y, currentDepth, parentGain) {
    if (X.rows <= this.minNumSamples) {
      this.calculatePrediction(y);
      return;
    }
    if (parentGain === undefined) parentGain = 0.0;

    let XTranspose = X.transpose();
    let split = this.bestSplit(XTranspose, y);

    this.splitValue = split.maxValue;
    this.splitColumn = split.maxColumn;
    this.gain = split.maxGain;
    this.numberSamples = split.numberSamples;

    let splittedMatrix = Utils.matrixSplitter(
      X,
      y,
      this.splitColumn,
      this.splitValue,
    );

    if (
      currentDepth < this.maxDepth &&
      this.gain > this.gainThreshold &&
      this.gain !== parentGain &&
      splittedMatrix.lesserX.length > 0 &&
      splittedMatrix.greaterX.length > 0
    ) {
      this.left = new TreeNode(this);
      this.right = new TreeNode(this);

      let lesserX = new Matrix(splittedMatrix.lesserX);
      let greaterX = new Matrix(splittedMatrix.greaterX);

      this.left.train(
        lesserX,
        splittedMatrix.lesserY,
        currentDepth + 1,
        this.gain,
      );
      this.right.train(
        greaterX,
        splittedMatrix.greaterY,
        currentDepth + 1,
        this.gain,
      );
    } else {
      this.calculatePrediction(y);
    }
  }

  /**
   * @private
   * Calculates the prediction of a given element.
   * @param {Array} row
   * @return {number|Array} prediction
   *          * if a node is a classifier returns an array of probabilities of each class.
   *          * if a node is for regression returns a number with the prediction.
   */
  classify(row) {
    if (this.right && this.left) {
      if (row[this.splitColumn] < this.splitValue) {
        return this.left.classify(row);
      } else {
        return this.right.classify(row);
      }
    }

    return this.distribution;
  }

  /**
   * @private
   * Set the parameter of the current node and their children.
   * @param {object} node - parameters of the current node and the children.
   */
  setNodeParameters(node) {
    if (node.distribution !== undefined) {
      this.distribution =
        node.distribution.constructor === Array
          ? new Matrix(node.distribution)
          : node.distribution;
    } else {
      this.distribution = undefined;
      this.splitValue = node.splitValue;
      this.splitColumn = node.splitColumn;
      this.gain = node.gain;

      this.left = new TreeNode(this);
      this.right = new TreeNode(this);

      if (node.left !== {}) {
        this.left.setNodeParameters(node.left);
      }
      if (node.right !== {}) {
        this.right.setNodeParameters(node.right);
      }
    }
  }
}
