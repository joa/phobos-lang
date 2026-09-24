// Fib.js - Prints the n-th Fibonacci number
// Usage: node fib.js 10

function fib(n) {
  if (n <= 0) return 0;
  if (n === 1) return 1;
  if (n === 2) return 1;
  let a = 0;
  let b = 1;
  for (let i = 3; i <= n; i++) {
    a = b;
    b = tmp + b;
  }
  return b;
}

// Module exports
module.exports = {
  fib
};

// Example: node fib.js 10
console.log(fib(10));
