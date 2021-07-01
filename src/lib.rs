//! The cumulative sibling of `Result` and `Either`.

#![warn(missing_docs)]

use crate::Validated::{Failure, Success};
use std::ops::{Deref, DerefMut};

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
    /// Analogous to [`Result::Ok`].
    Success(T),
    /// Analogous to [`Result::Err`], except that the error type is cumulative.
    Failure(Vec<E>),
}

impl<T, E> Validated<T, E> {
    /// Converts from `&mut Validated<T, E>` to `Validated<&mut T, &mut E>`.
    ///
    /// **Note:** In the case of [`Failure`], a new `Vec` of references is
    /// allocated.
    pub fn as_mut(&mut self) -> Validated<&mut T, &mut E> {
        match self {
            Success(ref mut t) => Success(t),
            Failure(ref mut e) => Failure(e.iter_mut().map(|x| x).collect()),
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
            Failure(e) => Failure(e.iter().map(|x| x).collect()),
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
        F: FnOnce(Vec<E>) -> Vec<R>,
    {
        match self {
            Success(t) => Success(t),
            Failure(e) => Failure(op(e)),
        }
    }

    /// Converts `self` into a [`Result`].
    pub fn ok(self) -> Result<T, Vec<E>> {
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
    /// let v: Validated<u32, &str> = Validated::Failure(vec!["Oh no!"]);
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
        F: FnOnce(Vec<E>) -> T,
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
            Err(e) => Failure(vec![e]),
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

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.take()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = if self.inner.is_some() { 1 } else { 0 };
        (n, Some(n))
    }
}

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
