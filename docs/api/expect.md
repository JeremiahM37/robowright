# expect

Every matcher waits: it retries until the physical world catches up or its timeout runs out, then fails with what
it saw. `expect(x).not_` inverts a matcher, `expect(x).always` turns it into an invariant checked on every step,
`expect.poll(fn)` asserts on a value a function returns, and `expect(block).to_pass()` retries a block of
assertions.

::: robowright.expect.expect

::: robowright.expect.Expectation

::: robowright.expect.condition
