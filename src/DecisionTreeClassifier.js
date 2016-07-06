"use strict";

var Utils = require("./Utils");
var Tree = require("./TreeNode");
var Matrix = require("ml-matrix");

var gainFunctions = {
    gini: Utils.giniGain
};

class DecisionTreeClassifier {
    
    constructor(options) {
        if(options === undefined) options = {};
        options.gainFunction = gainFunctions[options.gainFunction];
        if(options.gainFunction === undefined) options.gainFunction = gainFunctions["gini"];
        if(options.splitFunction === undefined) options.splitFunction = Utils.mean;
        if(options.minNumSamples === undefined) options.minNumSamples = 3;
        if(options.maxDepth === undefined) options.maxDepth = Infinity;

        options.kind = "classifier";
        this.options = options;
    }

    train(trainingSet, trainingLabels) {
        this.root = new Tree(this.options);
        if(!Matrix.isMatrix(trainingSet)) trainingSet = new Matrix(trainingSet);
        this.root.train(trainingSet, trainingLabels, 0);
    }

    predict(toPredict) {
        var predictions = new Array(toPredict.length);

        for(var i = 0; i < toPredict.length; ++i) {
            predictions[i] = this.root.classify(toPredict[i]).maxRowIndex(0)[1];
        }

        return predictions;
    }

    
}



module.exports = DecisionTreeClassifier;