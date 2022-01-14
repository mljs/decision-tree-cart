export default {
  input: 'src/index.js',
  output: {
    file: 'cart.js',
    format: 'cjs',
  },
  external: ['ml-matrix', 'ml-array-mean'],
};
