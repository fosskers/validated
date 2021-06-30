//! The cumulative sibling of `Result` and `Either`.

/// Similar to `Result`, but cumulative in its error type.
///
/// Consider that when using `collect` in a "Traversable" way to pull a single
/// `Result` out of an `Iterator` containing many `Result`s, it will fail on the
/// first `Err` and short-circuit the iteration. This is suboptimal if we wish
/// to be made aware of every failure that (would have) occurred.
///
/// ```
/// use validated::Validated;
///
/// let v: Vec<Result<(), ()>> = vec![Ok(()), Ok(()), Ok(())];
/// assert_eq!(Validated::Success(()), v.into_iter().collect());
///
/// let v: Vec<Result<(), usize>> = vec![Ok(()), Err(1), Ok(()), Err(2), Ok(())];
/// assert_eq!(Validated::Failure(vec![1,2]), v.into_iter().collect());
/// ```
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub enum Validated<T, E> {
    /// Analogous to `Result::Ok`.
    Success(T),
    /// Analogous to `Result::Err`, except that the error type is cumulative.
    Failure(Vec<E>),
}

impl<T, E> Validated<T, E> {
    /// Was a given `Validated` operation completely successful?
    pub fn is_success(&self) -> bool {
        matches!(self, Validated::Success(_))
    }

    /// Did a given `Validated` operation have at least one failure?
    pub fn is_failure(&self) -> bool {
        matches!(self, Validated::Failure(_))
    }
}

impl<T, E> From<Result<T, E>> for Validated<T, E> {
    fn from(r: Result<T, E>) -> Self {
        match r {
            Ok(t) => Validated::Success(t),
            Err(e) => Validated::Failure(vec![e]),
        }
    }
}
