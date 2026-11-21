//! Cooperative cancellation of long computations.
//!
//! Every computation checks for cancellation at bounded work intervals (each training iteration
//! and sampler block, each decoded sentence, each target position of the fast HMM, and every
//! progress event). When cancellation was requested, [`check`] unwinds with a [`Cancelled`]
//! payload (via `resume_unwind`, so no panic message is printed); callers that started the work
//! catch it with `std::panic::catch_unwind` and look for the payload with [`is_cancelled`].
//! Rayon carries the payload from worker threads to the thread that started the parallel
//! operation. Checking never changes a computed value.
//!
//! **Per-call state.** A caller that may run concurrently with others (the Python binding) opens
//! a [`Scope`] for each call: its own token, registered in a process-wide list while the call
//! runs. [`Scope::enter`] makes the token current on a thread, and
//! [`crate::pipeline::with_threads`] makes the current token current on every worker of the pool
//! it builds, so [`check`] anywhere in the call sees it. [`cancel_active`] sets the token of
//! every registered call: an interrupt stops all calls running at that moment, never a call
//! started afterwards, and one call finishing never clears another's request.
//!
//! Without a current token, [`check`] reads a process-wide flag ([`request`] / [`reset`]), the
//! previous behaviour, for callers that use no scopes.

use std::any::Any;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

static FLAG: AtomicBool = AtomicBool::new(false);
static ACTIVE: Mutex<Vec<Token>> = Mutex::new(Vec::new());

/// The cancellation state of one call.
pub type Token = Arc<AtomicBool>;

thread_local! {
    static CURRENT: RefCell<Option<Token>> = const { RefCell::new(None) };
}

/// Panic payload used to unwind a cancelled computation.
#[derive(Debug)]
pub struct Cancelled;

/// One call's cancellation token, registered for [`cancel_active`] until dropped.
#[derive(Debug)]
pub struct Scope {
    token: Token,
}

impl Scope {
    /// A fresh, not cancelled token, registered as active.
    pub fn new() -> Self {
        let token = Arc::new(AtomicBool::new(false));
        lock_active().push(token.clone());
        Scope { token }
    }

    /// The token (to make it current on other threads, see [`set_current`]).
    pub fn token(&self) -> Token {
        self.token.clone()
    }

    /// Whether this call was cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.token.load(Ordering::Relaxed)
    }

    /// Cancel this call only.
    pub fn cancel(&self) {
        self.token.store(true, Ordering::Relaxed);
    }

    /// Make the token current on this thread until the guard is dropped.
    pub fn enter(&self) -> Entered {
        Entered {
            prev: set_current(Some(self.token.clone())),
        }
    }
}

impl Default for Scope {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        lock_active().retain(|t| !Arc::ptr_eq(t, &self.token));
    }
}

/// Guard of [`Scope::enter`]: restores the thread's previous token.
#[derive(Debug)]
pub struct Entered {
    prev: Option<Token>,
}

impl Drop for Entered {
    fn drop(&mut self) {
        set_current(self.prev.take());
    }
}

fn lock_active() -> std::sync::MutexGuard<'static, Vec<Token>> {
    ACTIVE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Make `token` this thread's current token; returns the previous one.
pub fn set_current(token: Option<Token>) -> Option<Token> {
    CURRENT.with(|c| c.replace(token))
}

/// This thread's current token, if any.
pub fn current() -> Option<Token> {
    CURRENT.with(|c| c.borrow().clone())
}

/// Cancel every call that is active now (every registered [`Scope`]).
pub fn cancel_active() {
    for t in lock_active().iter() {
        t.store(true, Ordering::Relaxed);
    }
}

/// Ask running computations to stop at their next check: sets the process-wide flag (for
/// callers without a scope) and cancels every active scope.
pub fn request() {
    FLAG.store(true, Ordering::Relaxed);
    cancel_active();
}

/// Clear the process-wide flag (a scope's token is never cleared: open a new scope instead).
pub fn reset() {
    FLAG.store(false, Ordering::Relaxed);
}

/// Whether cancellation has been requested for the current call (its token, or without one the
/// process-wide flag).
#[inline]
pub fn is_requested() -> bool {
    CURRENT.with(|c| match &*c.borrow() {
        Some(t) => t.load(Ordering::Relaxed),
        None => FLAG.load(Ordering::Relaxed),
    })
}

/// Unwind with [`Cancelled`] if cancellation was requested.
#[inline]
pub fn check() {
    if is_requested() {
        std::panic::resume_unwind(Box::new(Cancelled));
    }
}

/// Whether a caught panic payload is a [`Cancelled`] unwind.
pub fn is_cancelled(payload: &(dyn Any + Send)) -> bool {
    payload.is::<Cancelled>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::catch_unwind;

    #[test]
    fn check_unwinds_only_when_requested() {
        let s = Scope::new();
        let _g = s.enter();
        assert!(catch_unwind(check).is_ok());
        s.cancel();
        let r = catch_unwind(check);
        assert!(is_cancelled(&*r.unwrap_err()));
    }

    #[test]
    fn scopes_are_independent_and_cancel_active_reaches_all() {
        let a = Scope::new();
        let b = Scope::new();
        // a call finishing (its scope dropped) does not clear another's cancellation
        let _ga = a.enter();
        a.cancel();
        assert!(is_requested() && !b.is_cancelled());
        cancel_active();
        assert!(b.is_cancelled());
        drop(_ga);
        drop(a);
        assert!(b.is_cancelled());
        // a scope opened after the cancellation is not affected
        let c = Scope::new();
        let _gc = c.enter();
        assert!(!is_requested());
        assert!(catch_unwind(check).is_ok());
        drop(b);
        assert!(!c.is_cancelled());
    }

    #[test]
    fn entered_restores_the_previous_token() {
        let a = Scope::new();
        let b = Scope::new();
        let ga = a.enter();
        {
            let _gb = b.enter();
            assert!(Arc::ptr_eq(&current().unwrap(), &b.token));
        }
        assert!(Arc::ptr_eq(&current().unwrap(), &a.token));
        drop(ga);
        assert!(current().is_none());
    }
}
