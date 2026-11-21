//! Cancellation lives in its own test binary: the flag is process-wide and would abort the
//! training runs of other tests running concurrently.

use eflomal_core::bitext::{Bitext, TextOptions};
use eflomal_core::cancel;
use eflomal_core::pipeline::{train, Directions, TrainConfig};

#[test]
fn cancellation_unwinds_and_the_next_run_works() {
    let text: String = (0..100)
        .map(|k| format!("a{} b ||| x y{}\n", k % 7, k % 5))
        .collect();
    let bitext = Bitext::from_joined_bytes(text.as_bytes(), &TextOptions::default());
    let cfg = TrainConfig {
        directions: Directions::Forward,
        ..TrainConfig::default()
    };
    cancel::request();
    let r = std::panic::catch_unwind(|| train(&bitext, &cfg, &|_| {}));
    cancel::reset();
    assert!(cancel::is_cancelled(&*r.unwrap_err()));
    assert!(train(&bitext, &cfg, &|_| {}).is_ok());
}
