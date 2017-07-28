# CART (Classification and regression trees)

  [![NPM version][npm-image]][npm-url]
  [![build status][travis-image]][travis-url]
  [![David deps][david-image]][david-url]
  [![npm download][download-image]][download-url]
  
Decision trees using CART implementation.

## Installation

`npm install --save ml-cart`

## Usage
### As a classifier

```js
import irisDataset from 'ml-dataset-iris';
import {DecisionTreeClassifier as DTClassifier} from 'ml-cart';

var trainingSet = irisDataset.getNumbers();
var predictions = irisDataset.getClasses().map(
    (elem) => irisDataset.getDistinctClasses().indexOf(elem)
);

var options = {
    gainFunction: 'gini',
    maxDepth: 10,
    minNumSamples: 3
};

var classifier = new DTClassifier(options);
classifier.train(trainingSet, predictions);
var result = classifier.predict(trainingSet);
```

### As a regression

```js
import {DecisionTreeRegression as DTRegression} from 'ml-cart';

var x = new Array(100);
var y = new Array(100);
var val = 0.0;
for (var i = 0; i < x.length; ++i) {
    x[i] = val;
    y[i] = Math.sin(x[i]);
    val += 0.01;
}

var reg = new DTRegression();
reg.train(x, y);
var estimations = reg.predict(x);
```

## [API documentation](http://mljs.github.io/decision-tree-cart/)

## License

  [MIT](./LICENSE)

[npm-image]: https://img.shields.io/npm/v/ml-cart.svg?style=flat-square
[npm-url]: https://npmjs.org/package/ml-cart
[travis-image]: https://img.shields.io/travis/mljs/decision-tree-cart/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/mljs/decision-tree-cart
[david-image]: https://img.shields.io/david/mljs/decision-tree-cart.svg?style=flat-square
[david-url]: https://david-dm.org/mljs/decision-tree-cart
[download-image]: https://img.shields.io/npm/dm/ml-cart.svg?style=flat-square
[download-url]: https://npmjs.org/package/ml-cart
