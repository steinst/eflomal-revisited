//! eflomal-core: pure-Rust Bayesian word alignment with collapsed Gibbs sampling, a clean-room
//! reimplementation of eflomal's models (no Python dependency).

pub mod bitext;
pub mod cancel;
pub mod corpus;
pub mod counts;
pub mod decode;
pub mod hmm;
pub mod io;
pub mod model;
pub mod oracle;
pub mod pipeline;
pub mod priors;
pub mod rng;
pub mod sampler;
pub mod segment;
pub mod segment_data;
pub mod symmetrize;
pub mod timing;

/// Crate version, taken from Cargo metadata.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_set() {
        assert!(!super::version().is_empty());
    }
}
