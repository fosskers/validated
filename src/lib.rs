//! The cumulative sibling of `Result` and `Either`.
//!
//! The [`Validated`] type has special `FromIterator` instances that enable
//! _all_ errors in a sequence to be reported, not just the first one.
//!
//! # Motivation
//!
//! We might think of [`Iterator::collect`] as being for consolidating the
//! result of some chained iteration into a concrete container, like `Vec`.
//!
//! ```
//! let v = vec![1,2,3];
//! assert_eq!(vec![2,4,6], v.into_iter().map(|n| n * 2).collect::<Vec<u32>>());
//! ```
//!
//! But `collect` isn't limited to this; it can be used to "fold" down into any
//! type you like, provided that it implements [`FromIterator`]. Consider the
//! effects of such an `impl` for `Result`:
//!
//! ```
//! let v: Vec<u32> = vec![1, 2, 3];
//! let r: Result<Vec<u32>, &str> = v
//!     .into_iter()
//!     .map(|n| if n % 2 == 0 { Err("Oh no!") } else { Ok(n * 2) })
//!     .collect();
//! assert_eq!(Err("Oh no!"), r);
//! ```
//!
//! The `Result` has been "interweaved" and pulled back out. Critically, this
//! `collect` call short-circuits; `n * 2` is never called for `3`, since the
//! `map` "fails" at `2`. This is useful when we require a sequence of IO
//! actions to all succeed and we wish to cancel remaining operations as soon
//! as any error occurs.
//!
//! But what if we don't want to short circuit? What if we want a report of all
//! the errors that occurred?
//!
//! ## Cumulative Errors and `Validated`
//!
//! Consider three cases where we'd want a report of all errors, not just the
//! first one:
//!
//! 1. Form input validation.
//! 2. Type checking.
//! 3. Concurrent IO.
//!
//! In the first case, if a user makes several input mistakes, it's the best
//! experience for them if all errors are reported at once so that they can make
//! their corrections in a single pass.
//!
//! In the second case, knowing only the first detected type error might not
//! actually be the site of the real issue. We need everything that's broken to
//! be reported so we can make the best decision of what to fix.
//!
//! In the third case, it may be that halting your entire concurrent job upon
//! detection of a single failure isn't appropriate. You might instead want
//! everything to finish as it can, and then collect a bundle of errors at the
//! end.
//!
//! The [`Validated`] type accomodates these use cases; it is a "cumulative `Result`".
//!
//! ```
//! use validated::Validated::{self, Success, Failure};
//!
//! let v = vec![Success(1), Validated::fail("No!"), Success(3), Validated::fail("Ack!")];
//! let r: Validated<Vec<u32>, &str> = Failure(vec!["No!", "Ack!"]);
//! assert_eq!(r, v.into_iter().collect());
//! ```
//!
//! # Features
//!
//! - `rayon`: Enable `FromParallelIterator` instances for `Validated`.
//!
//! # Resources
//!
//! - [Haskell: Validation][haskell]
//! - [Scala: Cats `Validated`][cats]
//!
//! [haskell]: https://hackage.haskell.org/package/validation
//! [cats]: https://typelevel.org/cats/datatypes/validated.html

#![warn(missing_docs)]

use crate::Validated::{Failure, Success};
use nonempty::NonEmpty;
use std::iter::FromIterator;
use std::ops::{Deref, DerefMut};

#[cfg(feature = "rayon")]
use rayon::iter::{FromParallelIterator, IntoParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use std::sync::Mutex;

/// Similar to [`Result`], but cumulative in its error type.
///
/// Consider that when using `collect` in a "Traversable" way to pull a single
/// `Result` out of an `Iterator` containing many `Result`s, it will fail on the
/// first `Err` and short-circuit the iteration. This is suboptimal if we wish
/// to be made aware of every failure that (would have) occurred.
///
/// ```
/// use validated::Validated;
///
/// let v = vec![Ok(1), Ok(2), Ok(3)];
/// let r: Validated<Vec<u32>, &str> = Validated::Success(vec![1, 2, 3]);
/// assert_eq!(r, v.into_iter().collect());
///
/// let v = vec![Ok(1), Err("Oh!"), Ok(2), Err("No!"), Ok(3)];
/// let r: Validated<Vec<u32>, &str> = Validated::Failure(vec!["Oh!", "No!"]);
/// assert_eq!(r, v.into_iter().collect());
/// ```
///
/// Naturally iterators of `Validated` values can be collected in a similar way:
///
/// ```
/// use validated::Validated::{self, Success, Failure};
///
/// let v = vec![Success(1), Success(2), Success(3)];
/// let r: Validated<Vec<u32>, &str> = Success(vec![1, 2, 3]);
/// assert_eq!(r, v.into_iter().collect());
///
/// let v = vec![Success(1), Validated::fail("No!"), Success(3), Validated::fail("Ack!")];
/// let r: Validated<Vec<u32>, &str> = Failure(vec!["No!", "Ack!"]);
/// assert_eq!(r, v.into_iter().collect());
/// ```
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub enum Validated<T, E> {
    /// Analogous to [`Result::Ok`].
    Success(T),
    /// Analogous to [`Result::Err`], except that the error type is cumulative.
    Failure(NonEmpty<E>),
}

impl<T, E> Validated<T, E> {
    /// Fail with the given error.
    pub fn fail(e: E) -> Validated<T, E> {
        Failure(NonEmpty::new(e))
    }

    /// Converts from `&mut Validated<T, E>` to `Validated<&mut T, &mut E>`.
    ///
    /// **Note:** In the case of [`Failure`], a new `Vec` of references is
    /// allocated.
    pub fn as_mut(&mut self) -> Validated<&mut T, &mut E> {
        match self {
            Success(ref mut t) => Success(t),
            Failure(ref mut e) => {
                let head = &mut e.head;
                let tail = e.tail.iter_mut().collect();
                let ne = NonEmpty { head, tail };
                Failure(ne)
            }
        }
    }

    /// Converts from `&Validated<T, E>` to `Validated<&T, &E>`.
    ///
    /// Produces a new `Validated`, containing references to the original,
    /// leaving the original in place.
    ///
    /// **Note:** In the case of [`Failure`], a new `Vec` of references is
    /// allocated.
    pub fn as_ref(&self) -> Validated<&T, &E> {
        match self {
            Success(ref t) => Success(t),
            Failure(e) => {
                let head = &e.head;
                let tail = e.tail.iter().collect();
                let ne = NonEmpty { head, tail };
                Failure(ne)
            }
        }
    }

    /// Returns the contained [`Success`] value, consuming `self`.
    ///
    /// # Panics
    ///
    /// Panics with a custom message if `self` is actually the `Failure`
    /// variant.
    pub fn expect(self, msg: &str) -> T {
        match self {
            Success(t) => t,
            Failure(_) => panic!("{}", msg),
        }
    }

    /// Was a given `Validated` operation completely successful?
    pub fn is_success(&self) -> bool {
        matches!(self, Success(_))
    }

    /// Did a given `Validated` operation have at least one failure?
    pub fn is_failure(&self) -> bool {
        matches!(self, Failure(_))
    }

    /// Returns an iterator over the possibly contained value.
    ///
    /// The iterator yields one value if the result is [`Success`], otherwise
    /// nothing.
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            inner: self.as_ref().ok().ok(),
        }
    }

    /// Returns a mutable iterator over the possibly contained value.
    ///
    /// The iterator yields one value if the result is [`Success`], otherwise
    /// nothing.
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut {
            inner: self.as_mut().ok().ok(),
        }
    }

    /// Applies a function to the `T` value of a [`Success`] variant, or leaves
    /// a [`Failure`] variant untouched.
    ///
    /// ```
    /// use validated::Validated::{self, Success, Failure};
    ///
    /// let v: Validated<u32, &str> = Success(1);
    /// let r = v.map(|n| n * 2);
    /// assert_eq!(r, Success(2));
    ///
    /// let v: Validated<u32, &str> = Validated::fail("No!");
    /// let r = v.map(|n| n * 2);
    /// assert_eq!(r, Validated::fail("No!"));
    /// ```
    pub fn map<U, F>(self, op: F) -> Validated<U, E>
    where
        F: FnOnce(T) -> U,
    {
        match self {
            Success(t) => Success(op(t)),
            Failure(e) => Failure(e),
        }
    }

    /// Applies a function to the `Vec<E>` of a [`Failure`] variant, or leaves a
    /// [`Success`] variant untouched.
    pub fn map_err<R, F>(self, op: F) -> Validated<T, R>
    where
        F: FnOnce(NonEmpty<E>) -> NonEmpty<R>,
    {
        match self {
            Success(t) => Success(t),
            Failure(e) => Failure(op(e)),
        }
    }

    /// Maps a function over two `Validated`, but only if both are of the
    /// `Success` variant. If both failed, then their errors are concatenated.
    pub fn map2<U, Z, F>(self, vu: Validated<U, E>, f: F) -> Validated<Z, E>
    where
        F: FnOnce(T, U) -> Z,
    {
        match (self, vu) {
            (Success(t), Success(u)) => Success(f(t, u)),
            (Success(_), Failure(e)) => Failure(e),
            (Failure(e), Success(_)) => Failure(e),
            (Failure(mut e0), Failure(mut e1)) => {
                e0.push(e1.head);
                e0.append(&mut e1.tail);
                Failure(e0)
            }
        }
    }

    /// Converts `self` into a [`Result`].
    pub fn ok(self) -> Result<T, NonEmpty<E>> {
        match self {
            Success(t) => Ok(t),
            Failure(e) => Err(e),
        }
    }

    /// Returns the contained [`Success`] value, consuming `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use validated::Validated;
    ///
    /// let v: Validated<u32, &str> = Validated::Success(1);
    /// assert_eq!(v.unwrap(), 1);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `self` is actually the `Failure` variant.
    pub fn unwrap(self) -> T {
        match self {
            Success(t) => t,
            Failure(_) => panic!("called `Validated::unwrap` on a `Failure` value"),
        }
    }

    /// Returns the contained [`Success`] value or a provided default.
    ///
    /// Arguments passed to `unwrap_or` are eagerly evaluated; if you are
    /// passing the result of a function call, it is recommended to use
    /// [`Validated::unwrap_or_else`] instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use validated::Validated;
    ///
    /// let v: Validated<u32, &str> = Validated::Success(1);
    /// assert_eq!(v.unwrap_or(2), 1);
    ///
    /// let v: Validated<u32, &str> = Validated::fail("Oh no!");
    /// assert_eq!(v.unwrap_or(2), 2);
    /// ```
    pub fn unwrap_or(self, default: T) -> T {
        match self {
            Success(t) => t,
            Failure(_) => default,
        }
    }

    /// Returns the contained [`Success`] value or computes it from a closure.
    pub fn unwrap_or_else<F>(self, op: F) -> T
    where
        F: FnOnce(NonEmpty<E>) -> T,
    {
        match self {
            Success(t) => t,
            Failure(e) => op(e),
        }
    }
}

impl<T: Default, E> Validated<T, E> {
    /// Returns the contained [`Success`] value or the default for `T`.
    pub fn unwrap_or_default(self) -> T {
        match self {
            Success(t) => t,
            Failure(_) => Default::default(),
        }
    }
}

impl<T: Deref, E> Validated<T, E> {
    /// Like [`Result::as_deref`].
    pub fn as_deref(&self) -> Validated<&T::Target, &E> {
        self.as_ref().map(|t| t.deref())
    }
}

impl<T: DerefMut, E> Validated<T, E> {
    /// Like [`Result::as_deref_mut`].
    pub fn as_deref_mut(&mut self) -> Validated<&mut T::Target, &mut E> {
        self.as_mut().map(|t| t.deref_mut())
    }
}

impl<T, E> From<Result<T, E>> for Validated<T, E> {
    fn from(r: Result<T, E>) -> Self {
        match r {
            Ok(t) => Success(t),
            Err(e) => Failure(NonEmpty::new(e)),
        }
    }
}

impl<T, U, E> FromIterator<Result<T, E>> for Validated<U, E>
where
    U: FromIterator<T>,
{
    fn from_iter<I: IntoIterator<Item = Result<T, E>>>(iter: I) -> Self {
        let mut errors = Vec::new();

        let result = iter
            .into_iter()
            .filter_map(|item| match item {
                Ok(t) => Some(t),
                Err(e) => {
                    errors.push(e);
                    None
                }
            })
            .collect();

        match NonEmpty::from_vec(errors) {
            None => Success(result),
            Some(e) => Failure(e),
        }
    }
}

impl<T, U, E> FromIterator<Validated<T, E>> for Validated<U, E>
where
    U: FromIterator<T>,
{
    fn from_iter<I: IntoIterator<Item = Validated<T, E>>>(iter: I) -> Self {
        let mut errors = Vec::new();

        let result = iter
            .into_iter()
            .filter_map(|item| match item {
                Success(t) => Some(t),
                Failure(e) => {
                    errors.extend(e);
                    None
                }
            })
            .collect();

        match NonEmpty::from_vec(errors) {
            None => Success(result),
            Some(e) => Failure(e),
        }
    }
}

#[cfg(feature = "rayon")]
impl<T, U, E> FromParallelIterator<Result<T, E>> for Validated<U, E>
where
    T: Send,
    E: Send,
    U: FromParallelIterator<T>,
{
    fn from_par_iter<I>(par_iter: I) -> Validated<U, E>
    where
        I: IntoParallelIterator<Item = Result<T, E>>,
    {
        let errors = Mutex::new(Vec::new());

        let result = par_iter
            .into_par_iter()
            .filter_map(|item| match item {
                Ok(t) => Some(t),
                Err(e) => {
                    errors.lock().unwrap().push(e);
                    None
                }
            })
            .collect();

        match NonEmpty::from_vec(errors.into_inner().unwrap()) {
            None => Success(result),
            Some(e) => Failure(e),
        }
    }
}

#[cfg(feature = "rayon")]
impl<T, U, E> FromParallelIterator<Validated<T, E>> for Validated<U, E>
where
    T: Send,
    E: Send,
    U: FromParallelIterator<T>,
{
    fn from_par_iter<I>(par_iter: I) -> Validated<U, E>
    where
        I: IntoParallelIterator<Item = Validated<T, E>>,
    {
        let errors = Mutex::new(Vec::new());

        let result = par_iter
            .into_par_iter()
            .filter_map(|item| match item {
                Success(t) => Some(t),
                Failure(e) => {
                    errors.lock().unwrap().extend(e);
                    None
                }
            })
            .collect();

        match NonEmpty::from_vec(errors.into_inner().unwrap()) {
            None => Success(result),
            Some(e) => Failure(e),
        }
    }
}

impl<T, E> IntoIterator for Validated<T, E> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.ok().ok(),
        }
    }
}

impl<'a, T, E> IntoIterator for &'a Validated<T, E> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T, E> IntoIterator for &'a mut Validated<T, E> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> IterMut<'a, T> {
        self.iter_mut()
    }
}

/// An iterator over a reference to the [`Success`] variant of a [`Validated`].
///
/// The iterator yields one value if the result is [`Success`], otherwise nothing.
///
/// Created by [`Validated::iter`].
#[derive(Debug)]
pub struct Iter<'a, T: 'a> {
    inner: Option<&'a T>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.take()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = if self.inner.is_some() { 1 } else { 0 };
        (n, Some(n))
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.take()
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {}

/// An iterator over a mutable reference to the [`Success`] variant of a [`Validated`].
///
/// Created by [`Validated::iter_mut`].
#[derive(Debug)]
pub struct IterMut<'a, T: 'a> {
    inner: Option<&'a mut T>,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<&'a mut T> {
        self.inner.take()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = if self.inner.is_some() { 1 } else { 0 };
        (n, Some(n))
    }
}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.take()
    }
}

impl<T> ExactSizeIterator for IterMut<'_, T> {}

/// An iterator over the value in a [`Success`] variant of a [`Validated`].
///
/// The iterator yields one value if the result is [`Success`], otherwise nothing.
///
/// This struct is created by the [`into_iter`] method on
/// [`Validated`] (provided by the [`IntoIterator`] trait).
///
/// [`into_iter`]: IntoIterator::into_iter
#[derive(Clone, Debug)]
pub struct IntoIter<T> {
    inner: Option<T>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.inner.take()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = if self.inner.is_some() { 1 } else { 0 };
        (n, Some(n))
    }
}
