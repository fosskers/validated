# validated

<!-- cargo-rdme start -->

The cumulative sibling of `Result` and `Either`.

The `Validated` type has special `FromIterator` instances that enable
_all_ errors in a sequence to be reported, not just the first one.

## Motivation

We might think of `Iterator::collect` as being for consolidating the
result of some chained iteration into a concrete container, like `Vec`.

```rust
let v = vec![1,2,3];
assert_eq!(vec![2,4,6], v.into_iter().map(|n| n * 2).collect::<Vec<u32>>());
```

But `collect` isn't limited to this; it can be used to "fold" down into any
type you like, provided that it implements `FromIterator`. Consider the
effects of such an `impl` for `Result`:

```rust
let v: Vec<u32> = vec![1, 2, 3];
let r: Result<Vec<u32>, &str> = v
    .into_iter()
    .map(|n| if n % 2 == 0 { Err("Oh no!") } else { Ok(n * 2) })
    .collect();
assert_eq!(Err("Oh no!"), r);
```

The `Result` has been "interweaved" and pulled back out. Critically, this
`collect` call short-circuits; `n * 2` is never called for `3`, since the
`map` "fails" at `2`. This is useful when we require a sequence of IO
actions to all succeed and we wish to cancel remaining operations as soon
as any error occurs.

But what if we don't want to short circuit? What if we want a report of all
the errors that occurred?

### Cumulative Errors and `Validated`

Consider three cases where we'd want a report of all errors, not just the
first one:

1. Form input validation.
2. Type checking.
3. Concurrent IO.

In the first case, if a user makes several input mistakes, it's the best
experience for them if all errors are reported at once so that they can make
their corrections in a single pass.

In the second case, knowing only the first detected type error might not
actually be the site of the real issue. We need everything that's broken to
be reported so we can make the best decision of what to fix.

In the third case, it may be that halting your entire concurrent job upon
detection of a single failure isn't appropriate. You might instead want
everything to finish as it can, and then collect a bundle of errors at the
end.

The `Validated` type accomodates these use cases; it is a "cumulative `Result`".

```rust
use validated::Validated::{self, Good, Fail};
use nonempty_collections::*;

let v = vec![Good(1), Validated::fail("No!"), Good(3), Validated::fail("Ack!")];
let r: Validated<Vec<u32>, &str> = Fail(nev!["No!", "Ack!"]);
assert_eq!(r, v.into_iter().collect());
```

### Use of non-empty Vectors (`NEVec`)

In the spirit of "make illegal states unrepresentable", the `Fail` variant
of `Validated` contains a `NEVec`, a non-empty `Vec`. `NEVec` can do
everything that `Vec` can do, plus some additional benefits. In the case of
this crate, this representation forbids the otherwise meaningless `Fail(vec![])`.

In other words, if you have a `Validated<T, E>`, you either have a concrete
`T`, or **at least one** `E`.

## Features

- `rayon`: Enable `FromParallelIterator` instances for `Validated`.

## Resources

- [Haskell: Validation][haskell]
- [Scala: Cats `Validated`][cats]

[haskell]: https://hackage.haskell.org/package/validation
[cats]: https://typelevel.org/cats/datatypes/validated.html

<!-- cargo-rdme end -->

