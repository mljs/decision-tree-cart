# ml-cart (Classification and regression trees)

[![NPM version][npm-image]][npm-url]
[![build status][ci-image]][ci-url]
[![npm download][download-image]][download-url]

Decision trees using CART implementation.

## Installation

`npm i ml-cart`

## [API documentation](http://mljs.github.io/decision-tree-cart/)

## Usage

### As a classifier

```js
import irisDataset from 'ml-dataset-iris';
import { DecisionTreeClassifier as DTClassifier } from 'ml-cart';

const trainingSet = irisDataset.getNumbers();
const predictions = irisDataset
  .getClasses()
  .map((elem) => irisDataset.getDistinctClasses().indexOf(elem));

const options = {
  gainFunction: 'gini',
  maxDepth: 10,
  minNumSamples: 3,
};

const classifier = new DTClassifier(options);
classifier.train(trainingSet, predictions);
const result = classifier.predict(trainingSet);
```

### As a regression

```js
import { DecisionTreeRegression as DTRegression } from 'ml-cart';

const x = new Array(100);
const y = new Array(100);
const val = 0.0;
for (let i = 0; i < x.length; ++i) {
  x[i] = val;
  y[i] = Math.sin(x[i]);
  val += 0.01;
}

const reg = new DTRegression();
reg.train(x, y);
const estimations = reg.predict(x);
```

## License

[MIT](./LICENSE)

[npm-image]: https://img.shields.io/npm/v/ml-cart.svg
[npm-url]: https://npmjs.org/package/ml-cart
[ci-image]: https://github.com/mljs/decision-tree-cart/workflows/Node.js%20CI/badge.svg?branch=master
[ci-url]: https://github.com/mljs/decision-tree-cart/actions?query=workflow%3A%22Node.js+CI%22
[download-image]: https://img.shields.io/npm/dm/ml-cart.svg
[download-url]: https://npmjs.org/package/ml-cart
