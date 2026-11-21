//! Degenerate inputs: training, saving, loading and aligning must neither panic nor
//! give output of the wrong shape, and the defined behaviour of each edge case holds.

use eflomal_core::bitext::{Bitext, Parsed, TextOptions};
use eflomal_core::io::{Direction, Model};
use eflomal_core::pipeline::{
    align, align_fast, corpus_warnings, train, with_threads, AlignConfig, Directions, TrainConfig,
    Trained,
};
use eflomal_core::segment::Segment;

/// Joined (`src ||| tgt`) inputs.
const JOINED: &[&[u8]] = &[
    b"",
    b"\n",
    b"\n\n\n",
    b" ||| \n",
    b"a ||| x\n",
    b"a ||| \n",
    b" ||| x\n",
    b"<NULL> ||| <NULL>\n",
    b"a b c d ||| x\n",
    b"a b ||| x y\r\nc ||| z",
    b"a ||| b ||| c\nno separator\n\t\n",
    b"\xe9 \xff ||| \x00 \xfe\n",
    b"a\rb ||| x\ry\n",
    "дом ||| σοφός ΣΟΦΟΣ\n我 的 ||| بيتي\n".as_bytes(),
    b"the house ||| la maison\nthe blue house ||| la maison bleue\na book ||| un livre\n",
];

fn configs() -> Vec<TrainConfig> {
    let mut v = Vec::new();
    for (model, samplers, iters, exact, scores) in [
        (3, 3, [Some(2), Some(2), Some(3)], false, true),
        (1, 1, [Some(1), None, None], false, false),
        (2, 2, [Some(0), Some(0), None], true, false),
        (3, 8, [Some(0), Some(0), Some(0)], false, true),
        (3, 2, [Some(1), Some(0), Some(2)], true, true),
    ] {
        v.push(TrainConfig {
            model,
            samplers,
            iters,
            exact,
            scores,
            keep_model: true,
            ..TrainConfig::default()
        });
    }
    v
}

fn n_lines(t: &[u8]) -> usize {
    t.iter().filter(|&&b| b == b'\n').count()
}

/// Every link is inside its pair, every output has one line per input line.
fn check_shape(bitext: &Bitext, text: &[u8], d: Direction) {
    assert_eq!(n_lines(text), bitext.len(), "{d:?}");
    let v = bitext.view(Direction::Forward);
    for (k, line) in text.split(|&b| b == b'\n').take(bitext.len()).enumerate() {
        let links = line.split(|&b| b == b'|').next().unwrap().trim_ascii();
        let (s, t) = v.pair(k);
        if s.is_empty() || t.is_empty() {
            assert!(links.is_empty(), "line {k} has an empty side but links");
        }
        for l in links.split(|&b| b == b' ').filter(|l| !l.is_empty()) {
            let l = std::str::from_utf8(l).unwrap();
            let (i, j) = l.split_once('-').unwrap();
            let (i, j): (usize, usize) = (i.parse().unwrap(), j.parse().unwrap());
            assert!(
                i < s.len() && j < t.len(),
                "line {k}: link {l} out of range"
            );
        }
    }
}

fn train_t(b: &Bitext, cfg: &TrainConfig, threads: usize) -> Trained {
    with_threads(threads, || train(b, cfg, &|_| {})).unwrap()
}

#[test]
fn degenerate_corpora_train_save_load_align() {
    for &text in JOINED {
        let b = Parsed::from_joined_bytes(text, None)
            .build_segmented(&TextOptions::default(), Segment::Auto);
        for cfg in configs() {
            let out = train_t(&b, &cfg, 2);
            for d in [Direction::Forward, Direction::Inverse] {
                check_shape(&b, out.output(d).unwrap(), d);
            }
            // the same at another thread count
            let again = train_t(&b, &cfg, 3);
            assert_eq!(out.fwd, again.fwd);
            assert_eq!(out.inv, again.inv);
            let model = out.model.as_ref().unwrap();
            let mut bytes = Vec::new();
            model.write(&mut bytes).unwrap();
            let loaded = Model::read(&bytes[..], bytes.len() as u64).unwrap();
            assert_eq!(&loaded, model);
            let nb = Parsed::from_joined_bytes(text, None).build(&loaded.options.text);
            let a = AlignConfig {
                iters: [Some(1), Some(1), Some(1)],
                ..AlignConfig::default()
            };
            let al = align(&loaded, &nb, &a, &|_| {}).unwrap();
            let fa = align_fast(&loaded, &nb, None).unwrap();
            for d in [Direction::Forward, Direction::Inverse] {
                check_shape(&nb, al.output(d).unwrap(), d);
                check_shape(&nb, fa.output(d).unwrap(), d);
            }
        }
    }
}

#[test]
fn nothing_to_align_warns_and_succeeds() {
    let cases: &[(&[u8], usize, &str)] = &[
        (b"", 0, "no line pairs"),
        (b"\n\n", 2, "none of the 2 input line pairs"),
        (b" ||| x\na |||\n", 2, "none of the 2"),
        (b"a ||| x\n", 1, ""),
        (b" ||| \n", 1, "the only input line pair does not have"),
    ];
    for &(text, k, want) in cases {
        let b = Bitext::from_joined_bytes(text, &TextOptions::default());
        let w = corpus_warnings(&b);
        if want.is_empty() {
            assert!(w.is_empty(), "{w:?}");
        } else {
            assert!(w.len() == 1 && w[0].contains(want), "{w:?}");
        }
        let out = train_t(&b, &TrainConfig::default(), 2);
        assert_eq!(n_lines(out.fwd.as_ref().unwrap()), k);
    }
    // over --max-len counts as no words
    let opts = TextOptions {
        max_len: 2,
        ..TextOptions::default()
    };
    let b = Bitext::from_joined_bytes(b"a b c ||| x\n", &opts);
    assert_eq!(b.skipped(), [0]);
    assert!(corpus_warnings(&b)[0].contains("within --max-len"));
}

fn fwd_of(src: &[u8], tgt: &[u8], cfg: &TrainConfig) -> Vec<u8> {
    let b = Parsed::from_pair_bytes(src, tgt, None)
        .unwrap()
        .build_segmented(&TextOptions::default(), Segment::Auto);
    train_t(&b, cfg, 2).fwd.unwrap()
}

#[test]
fn line_ends_do_not_change_the_output() {
    let cfg = TrainConfig {
        directions: Directions::Forward,
        iters: [Some(3), Some(3), Some(3)],
        ..TrainConfig::default()
    };
    let s = b"the house\nthe blue house\na book\n";
    let t = b"la maison\nla maison bleue\nun livre\n";
    let lf = fwd_of(s, t, &cfg);
    // CRLF, and no final newline, give the same bytes
    let crlf = fwd_of(
        b"the house\r\nthe blue house\r\na book\r\n",
        b"la maison\r\nla maison bleue\r\nun livre\r\n",
        &cfg,
    );
    assert_eq!(lf, crlf);
    let open = fwd_of(&s[..s.len() - 1], &t[..t.len() - 1], &cfg);
    assert_eq!(lf, open);
    // a lone CR is whitespace, not a line end (eflomal's Python reader would split there)
    let b = Bitext::from_pair_bytes(b"a\rb\n", b"x\n", &TextOptions::default()).unwrap();
    assert_eq!((b.len(), b.source.line_len(0)), (1, 2));
}

#[test]
fn bytes_that_are_not_utf8_align() {
    // eflomal crashes on Latin-1 input; tokens are bytes here
    let b = Bitext::from_pair_bytes(
        b"caf\xe9 \xe9t\xe9\nla \xff\n",
        b"coffee summer\nthe x\n",
        &TextOptions::default(),
    )
    .unwrap();
    assert_eq!(b.source.n_types(), 5); // NULL + café été la ÿ (ASCII-lowercased bytes)
    let out = train_t(&b, &TrainConfig::default(), 2);
    check_shape(&b, out.fwd.as_ref().unwrap(), Direction::Forward);
}

#[test]
fn zero_stem_length_means_no_stemming() {
    let s = b"houses housing\nbooks booking\n";
    let t = b"maisons logement\nlivres reservation\n";
    let words = Parsed::from_pair_bytes(s, t, None)
        .unwrap()
        .build_segmented(&TextOptions::default(), Segment::Words);
    for zero in [
        TextOptions {
            source_prefix: Some(0),
            ..TextOptions::default()
        },
        TextOptions {
            source_suffix: Some(0),
            target_prefix: Some(0),
            ..TextOptions::default()
        },
    ] {
        let b = Parsed::from_pair_bytes(s, t, None)
            .unwrap()
            .build_segmented(&zero, Segment::Auto);
        // explicit (auto is off), but nothing is stemmed and nothing 0 is stored
        assert_eq!(b.options(), words.options());
        assert_eq!(b.source.n_types(), words.source.n_types());
        let cfg = TrainConfig {
            keep_model: true,
            ..TrainConfig::default()
        };
        let (x, y) = (train_t(&b, &cfg, 2), train_t(&words, &cfg, 2));
        assert_eq!(x.fwd, y.fwd);
        assert_eq!(x.model.unwrap().options.text, y.model.unwrap().options.text);
    }
}

#[test]
fn length_zero_is_the_minimum_schedule() {
    let b = Bitext::from_joined_bytes(b"a ||| x\n", &TextOptions::default());
    let cfg = TrainConfig {
        length: 0.0,
        ..TrainConfig::default()
    };
    assert_eq!(train_t(&b, &cfg, 1).iterations, [2, 1, 2]);
}

#[test]
fn long_pairs_beyond_eflomals_limit() {
    let long: String = (0..1500).map(|i| format!("w{} ", i % 300)).collect();
    let text = format!("a b ||| x y\n{long}||| {long}\nb ||| y\n");
    let cfg = TrainConfig {
        iters: [Some(2), Some(2), Some(2)],
        samplers: 1,
        ..TrainConfig::default()
    };
    // default --max-len 1023: the long pair is skipped and gets an empty line
    let b = Bitext::from_joined_bytes(text.as_bytes(), &TextOptions::default());
    assert_eq!(b.skipped(), [1]);
    let out = train_t(&b, &cfg, 2);
    let fwd = out.fwd.unwrap();
    assert_eq!(fwd.split(|&c| c == b'\n').nth(1).unwrap(), b"");
    // a larger --max-len aligns it
    let opts = TextOptions {
        max_len: 2000,
        ..TextOptions::default()
    };
    let b = Bitext::from_joined_bytes(text.as_bytes(), &opts);
    assert!(b.skipped().is_empty());
    let out = train_t(&b, &cfg, 2);
    check_shape(&b, out.fwd.as_ref().unwrap(), Direction::Forward);
    let line1 = out
        .fwd
        .unwrap()
        .split(|&c| c == b'\n')
        .nth(1)
        .unwrap()
        .to_vec();
    assert!(!line1.is_empty());
}

#[test]
fn heavy_duplication_and_more_samplers_than_threads() {
    let text = "a b c ||| x y z\n".repeat(3000);
    let b = Bitext::from_joined_bytes(text.as_bytes(), &TextOptions::default());
    let cfg = TrainConfig {
        samplers: 8,
        iters: [Some(2), Some(2), Some(4)],
        ..TrainConfig::default()
    };
    let one = train_t(&b, &cfg, 1);
    let two = train_t(&b, &cfg, 2);
    assert_eq!(one.fwd, two.fwd);
    assert_eq!(one.inv, two.inv);
    // identical pairs: every line gets the same links
    let fwd = one.fwd.unwrap();
    let first = fwd.split(|&c| c == b'\n').next().unwrap().to_vec();
    assert!(!first.is_empty());
    assert!(fwd
        .split(|&c| c == b'\n')
        .take(3000)
        .all(|l| l == &first[..]));
}
