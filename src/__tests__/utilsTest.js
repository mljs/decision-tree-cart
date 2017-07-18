import * as Utils from '../utils';

describe('Utils', () => {
    test('Gini gain', () => {
        expect(Utils.giniGain([0, 1, 0, 1, 0, 1], {
            greater: [0, 0],
            lesser: [1, 1, 1, 0]
        })).toBeCloseTo(0.25, 3);
    });

    test('Regression error', () => {
        var y = [0.5, 0.7, 0.8, 0.9, 1, 1.1];
        var splitted = {
            greater: [0.5, 0.7],
            lesser: [0.8, 0.9, 1.0, 1.1]
        };

        expect(Utils.regressionError(y, splitted)).toBeCloseTo(0.07, 2);
    });

    test('Get number of classes', () => {
        expect(Utils.getNumberOfClasses([0, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9])).toBe(10);
    });
});
