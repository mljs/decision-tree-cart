import {Matrix as Matrix} from 'ml-matrix';
import Tree from './TreeNode';

export default class DecisionTreeClassifier {

    /**
     * Create new Decision Tree Classifier with CART implementation with the given options
     * @param {object} options
     * @param {string} [options.gainFunction="gini"] - gain function to get the best split, "gini" the only one supported.
     * @param {string} [options.splitFunction] - given two integers from a split feature, get the value to split, "mean" the only one supported.
     * @param {number} [options.minNumSamples] - minimum number of samples to create a leaf node to decide a class. Default 3.
     * @param {number} [options.maxDepth] - Max depth of the tree. Default Infinity.
     * @param {object} model - for load purposes.
     * @constructor
     */
    constructor(options, model) {
        if (options === true) {
            this.options = model.options;
            this.root = new Tree(model.options);
            this.root.setNodeParameters(model.root);
        } else {
            if (options === undefined) options = {};
            if (options.gainFunction === undefined) options.gainFunction = 'gini';
            if (options.splitFunction === undefined) options.splitFunction = 'mean';
            if (options.minNumSamples === undefined) options.minNumSamples = 3;
            if (options.maxDepth === undefined) options.maxDepth = Infinity;

            options.kind = 'classifier';
            this.options = options;
        }
    }

    /**
     * Train the decision tree with the given training set and labels.
     * @param {Matrix} trainingSet
     * @param {Array} trainingLabels
     */
    train(trainingSet, trainingLabels) {
        this.root = new Tree(this.options);
        if (!Matrix.isMatrix(trainingSet)) trainingSet = new Matrix(trainingSet);
        this.root.train(trainingSet, trainingLabels, 0, null);
    }

    /**
     * Predicts the output given the matrix to predict.
     * @param {Matrix} toPredict
     * @return {Array} predictions
     */
    predict(toPredict) {
        var predictions = new Array(toPredict.length);

        for (var i = 0; i < toPredict.length; ++i) {
            predictions[i] = this.root.classify(toPredict[i]).maxRowIndex(0)[1];
        }

        return predictions;
    }

    /**
     * Export the current model to JSON.
     * @return {object} - Current model.
     */
    toJSON() {
        var toSave = {
            options: this.options,
            root: {},
            name: 'DTClassifier'
        };

        toSave.root = this.root;
        return toSave;
    }

    /**
     * Load a Decision tree classifier with the given model.
     * @param {object} model
     * @return {DecisionTreeClassifier}
     */
    static load(model) {
        if (model.name !== 'DTClassifier') {
            throw new RangeError('Invalid model: ' + model.name);
        }

        return new DecisionTreeClassifier(true, model);
    }
}
