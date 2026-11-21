//! Per-call cancellation scopes, in their own test binary: `cancel::request`
//! in other tests would cancel every active scope.

use std::panic::catch_unwind;

use eflomal_core::bitext::{Bitext, TextOptions};
use eflomal_core::cancel::{self, Scope};
use eflomal_core::pipeline::{train, with_threads, Directions, TrainConfig};

/// `cancel_active` reaches every scope in the process: the tests here take turns.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn corpus() -> Bitext {
    let text: String = (0..300)
        .map(|k| format!("a{} b c{} ||| x y{} z\n", k % 7, k % 3, k % 5))
        .collect();
    Bitext::from_joined_bytes(text.as_bytes(), &TextOptions::default())
}

#[test]
fn a_scope_reaches_pool_workers_and_spares_later_calls() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let b = corpus();
    let cfg = TrainConfig {
        directions: Directions::Forward,
        ..TrainConfig::default()
    };
    for threads in [0, 1, 3] {
        // cancelled from inside a progress event (as a raising Python callback does): the
        // workers of the call's pool see the call's token
        let s = Scope::new();
        let g = s.enter();
        let r = catch_unwind(|| {
            with_threads(threads, || train(&b, &cfg, &|_| cancel::cancel_active()))
        });
        assert!(cancel::is_cancelled(&*r.unwrap_err()), "{threads} threads");
        drop(g);
        // the cancelled call's scope is still open (still unwinding elsewhere); a new call runs
        let later = Scope::new();
        let g2 = later.enter();
        assert!(with_threads(threads, || train(&b, &cfg, &|_| {})).is_ok());
        assert!(s.is_cancelled() && !later.is_cancelled());
        drop(g2);
    }
}

#[test]
fn another_calls_cancellation_reaches_a_running_call() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let b = corpus();
    let cfg = TrainConfig {
        directions: Directions::Forward,
        ..TrainConfig::default()
    };
    let other = Scope::new();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::scope(|sc| {
        let h = sc.spawn(|| {
            let s = Scope::new();
            let _g = s.enter();
            let tx = std::sync::Mutex::new(Some(tx));
            catch_unwind(std::panic::AssertUnwindSafe(|| {
                with_threads(2, || {
                    train(&b, &cfg, &|_| {
                        if let Some(t) = tx.lock().unwrap().take() {
                            let _ = t.send(());
                        }
                        std::thread::sleep(std::time::Duration::from_millis(20));
                    })
                })
            }))
        });
        rx.recv().unwrap(); // the call is running
        drop(other); // a call finishing never clears another call's state ...
        cancel::cancel_active(); // ... and an interrupt reaches every active call
        let r = h.join().unwrap();
        assert!(cancel::is_cancelled(&*r.unwrap_err()));
    });
}
