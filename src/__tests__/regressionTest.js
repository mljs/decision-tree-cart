import { Matrix, MatrixTransposeView } from 'ml-matrix';

import { DecisionTreeRegression as DTRegression } from '..';

let x = new Array(100);
let y = new Array(100);
let val = 0.0;
for (let i = 0; i < x.length; ++i) {
  x[i] = val;
  y[i] = Math.sin(x[i]);
  val += 0.01;
}

let reg = new DTRegression();
reg.train(x, y);
let estimations = reg.predict(x);

describe('Decision tree regression', () => {
  it('Decision Tree classifier with sin function', () => {
    for (let i = 0; i < x.length; ++i) {
      expect(estimations[i]).toBeCloseTo(y[i], 0);
    }
  });

  it('Export and import for decision tree classifier', () => {
    let model = JSON.parse(JSON.stringify(reg));

    let newClassifier = DTRegression.load(model);
    let newEstimations = newClassifier.predict(x);

    expect(newEstimations).toStrictEqual(estimations);
  });

  it('Check transpose view', () => {
    x = Matrix.rowVector(x);
    x = new MatrixTransposeView(x);

    let output = reg.predict(x);
    for (let i = 0; i < output.length; ++i) {
      expect(output[i]).toBeCloseTo(y[i], 0);
    }
  });
});
