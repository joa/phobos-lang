// Run by scripts/agent_bench.py in the agent's copy of this folder, which does
// not contain this file. Exits non-zero unless fib(n) is right for 1..30.
const path = require("path");
const { fib } = require(path.resolve("fib.js"));
let a = 0, b = 1;
for (let n = 1; n <= 30; n++) {
  [a, b] = [b, a + b];
  if (fib(n) !== a) {
    console.error(`fib(${n}) = ${fib(n)}, want ${a}`);
    process.exit(1);
  }
}
