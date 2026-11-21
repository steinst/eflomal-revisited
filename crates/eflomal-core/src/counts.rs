//! Sparse integer count tables of the Gibbs sampler.
//!
//! * [`LexCounts`]: c(e, f) and n(e) = Σ_f c(e, f) for every source type e (id 0 = NULL, ε).
//!   Only non-zero counts are stored. A source row with at most [`INLINE`] entries keeps them
//!   inline in its row header; a larger row is an open-addressing table (linear probing,
//!   Fibonacci hashing, power-of-two capacity, load ≤ 1/2, shrinking below 1/8, deletion by
//!   backward shift, so there are no tombstones). The NULL row touches most target types and is
//!   a dense array.
//! * [`JumpCounts`]: d(m) over the buckets m ∈ −[`MAX_JUMP`] … [`MAX_JUMP`] (longer jumps are
//!   clamped) and their total D.
//! * [`FertCounts`]: c″(e, n) for fertilities n ∈ 0 … [`F_MAX`] (n ≥ F_MAX counted as F_MAX).
//!
//! Counts are `u32`. Lookups never depend on the order of the updates that produced the
//! counts: two tables with the same contents compare equal ([`LexCounts::same_counts`]) even if
//! their internal layouts differ.

/// Source id of the NULL word ε in every direction's source vocabulary.
pub const NULL: u32 = 0;
/// Largest absolute jump with its own bucket; longer jumps are clamped (M_J).
pub const MAX_JUMP: i32 = 64;
/// Number of jump buckets, −MAX_JUMP … MAX_JUMP.
pub const JUMP_BUCKETS: usize = 2 * MAX_JUMP as usize + 1;
/// Largest fertility category; it means "F_MAX or more".
pub const F_MAX: usize = 31;
/// Number of fertility categories, 0 … F_MAX.
pub const FERT_CATS: usize = F_MAX + 1;
/// Largest number of entries a row keeps inline, without a hash table.
pub const INLINE: usize = 4;

const EMPTY: u32 = u32::MAX;
const MIN_TABLE: usize = 16;

#[derive(Clone, Copy, Debug)]
struct Slot {
    key: u32,
    val: u32,
}

const EMPTY_SLOT: Slot = Slot { key: EMPTY, val: 0 };

/// One source row: inline entries or an open-addressing table.
#[derive(Clone, Debug)]
enum Row {
    Inline {
        n: u8,
        keys: [u32; INLINE],
        vals: [u32; INLINE],
    },
    Table {
        slots: Box<[Slot]>,
        len: u32,
        /// 32 − log2(capacity), for Fibonacci hashing.
        shift: u8,
    },
}

const EMPTY_ROW: Row = Row::Inline {
    n: 0,
    keys: [EMPTY; INLINE],
    vals: [0; INLINE],
};

#[inline]
fn home(key: u32, shift: u8) -> usize {
    (key.wrapping_mul(0x9E37_79B9) >> shift) as usize
}

fn new_table(cap: usize) -> (Box<[Slot]>, u8) {
    debug_assert!(cap.is_power_of_two() && cap >= MIN_TABLE);
    (
        vec![EMPTY_SLOT; cap].into_boxed_slice(),
        (32 - cap.trailing_zeros()) as u8,
    )
}

/// Insert a key known to be absent into a table with room.
#[inline]
fn insert_absent(slots: &mut [Slot], shift: u8, key: u32, val: u32) {
    let mask = slots.len() - 1;
    let mut i = home(key, shift);
    while slots[i].key != EMPTY {
        i = (i + 1) & mask;
    }
    slots[i] = Slot { key, val };
}

fn rehash(old: &[Slot], cap: usize) -> (Box<[Slot]>, u8) {
    let (mut slots, shift) = new_table(cap);
    for s in old.iter().filter(|s| s.key != EMPTY) {
        insert_absent(&mut slots, shift, s.key, s.val);
    }
    (slots, shift)
}

impl Row {
    #[inline]
    fn get(&self, key: u32) -> u32 {
        match self {
            Row::Inline { n, keys, vals } => {
                for k in 0..*n as usize {
                    if keys[k] == key {
                        return vals[k];
                    }
                }
                0
            }
            Row::Table { slots, shift, .. } => {
                let mask = slots.len() - 1;
                let mut i = home(key, *shift);
                loop {
                    let s = slots[i];
                    if s.key == key {
                        return s.val;
                    }
                    if s.key == EMPTY {
                        return 0;
                    }
                    i = (i + 1) & mask;
                }
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Row::Inline { n, .. } => *n as usize,
            Row::Table { len, .. } => *len as usize,
        }
    }

    /// Add `delta` to the count of `key`. Returns the change in the number of entries
    /// (+1 inserted, −1 removed, 0 otherwise). Panics if the count would become negative.
    fn add(&mut self, key: u32, delta: i64) -> isize {
        debug_assert!(key != EMPTY);
        if delta == 0 {
            return 0;
        }
        match self {
            Row::Inline { n, keys, vals } => {
                let len = *n as usize;
                if let Some(k) = keys[..len].iter().position(|&x| x == key) {
                    let v = i64::from(vals[k]) + delta;
                    let v = check_count(v, key);
                    if v == 0 {
                        keys[k] = keys[len - 1];
                        vals[k] = vals[len - 1];
                        keys[len - 1] = EMPTY;
                        vals[len - 1] = 0;
                        *n -= 1;
                        return -1;
                    }
                    vals[k] = v;
                    return 0;
                }
                let v = check_count(delta, key);
                if len < INLINE {
                    keys[len] = key;
                    vals[len] = v;
                    *n += 1;
                    return 1;
                }
                // Spill into a table.
                let (mut slots, shift) = new_table(MIN_TABLE);
                for k in 0..INLINE {
                    insert_absent(&mut slots, shift, keys[k], vals[k]);
                }
                insert_absent(&mut slots, shift, key, v);
                *self = Row::Table {
                    slots,
                    len: INLINE as u32 + 1,
                    shift,
                };
                1
            }
            Row::Table { slots, len, shift } => {
                let mask = slots.len() - 1;
                let mut i = home(key, *shift);
                loop {
                    let s = slots[i];
                    if s.key == key {
                        let v = check_count(i64::from(s.val) + delta, key);
                        if v != 0 {
                            slots[i].val = v;
                            return 0;
                        }
                        remove_at(slots, i);
                        *len -= 1;
                        if (*len as usize) * 8 < slots.len() {
                            self.shrink();
                        }
                        return -1;
                    }
                    if s.key == EMPTY {
                        break;
                    }
                    i = (i + 1) & mask;
                }
                let v = check_count(delta, key);
                let cap = slots.len();
                if (*len as usize + 1) * 2 > cap {
                    let (s2, sh2) = rehash(slots, cap * 2);
                    *slots = s2;
                    *shift = sh2;
                    insert_absent(slots, *shift, key, v);
                } else {
                    slots[i] = Slot { key, val: v };
                }
                *len += 1;
                1
            }
        }
    }

    /// Below load 1/8: go inline if the entries fit, else halve the table.
    fn shrink(&mut self) {
        let Row::Table { slots, len, shift } = self else {
            return;
        };
        let (l, cap) = (*len as usize, slots.len());
        if l <= INLINE {
            let mut keys = [EMPTY; INLINE];
            let mut vals = [0; INLINE];
            for (k, s) in slots.iter().filter(|s| s.key != EMPTY).enumerate() {
                keys[k] = s.key;
                vals[k] = s.val;
            }
            *self = Row::Inline {
                n: l as u8,
                keys,
                vals,
            };
        } else if cap > MIN_TABLE {
            let (s2, sh2) = rehash(slots, cap / 2);
            *slots = s2;
            *shift = sh2;
        }
    }

    fn for_each(&self, mut f: impl FnMut(u32, u32)) {
        match self {
            Row::Inline { n, keys, vals } => {
                for k in 0..*n as usize {
                    f(keys[k], vals[k]);
                }
            }
            Row::Table { slots, .. } => {
                for s in slots.iter().filter(|s| s.key != EMPTY) {
                    f(s.key, s.val);
                }
            }
        }
    }

    fn heap_bytes(&self) -> usize {
        match self {
            Row::Inline { .. } => 0,
            Row::Table { slots, .. } => slots.len() * std::mem::size_of::<Slot>(),
        }
    }
}

/// Remove slot `i` of a linear-probing table by backward shift.
fn remove_at(slots: &mut [Slot], mut i: usize) {
    let mask = slots.len() - 1;
    let shift = (32 - slots.len().trailing_zeros()) as u8;
    let mut j = i;
    loop {
        j = (j + 1) & mask;
        let s = slots[j];
        if s.key == EMPTY {
            break;
        }
        let h = home(s.key, shift);
        // Move s back to i if its home is not cyclically in (i, j].
        let in_range = if i <= j {
            i < h && h <= j
        } else {
            i < h || h <= j
        };
        if !in_range {
            slots[i] = s;
            i = j;
        }
    }
    slots[i] = EMPTY_SLOT;
}

#[inline]
#[track_caller]
fn check_count(v: i64, key: u32) -> u32 {
    assert!(
        (0..=i64::from(u32::MAX)).contains(&v),
        "count of target {key} out of range: {v}"
    );
    v as u32
}

/// Lexical counts c(e, f) and row totals n(e) of one sampler and direction.
#[derive(Clone, Debug)]
pub struct LexCounts {
    /// Rows of the source types 1 …; index 0 (NULL) is unused, see `null_row`.
    rows: Vec<Row>,
    totals: Vec<u32>,
    null_row: Vec<u32>,
    /// Stored entries in `rows` (non-zero counts outside the NULL row).
    entries: usize,
    /// Non-zero counts in the NULL row.
    null_entries: usize,
}

impl LexCounts {
    /// Empty counts for `n_source` source ids (including NULL = 0) and `n_target` target ids.
    pub fn new(n_source: usize, n_target: usize) -> Self {
        assert!(n_source >= 1, "the source vocabulary includes NULL");
        LexCounts {
            rows: vec![EMPTY_ROW; n_source],
            totals: vec![0; n_source],
            null_row: vec![0; n_target],
            entries: 0,
            null_entries: 0,
        }
    }

    pub fn n_source(&self) -> usize {
        self.rows.len()
    }

    pub fn n_target(&self) -> usize {
        self.null_row.len()
    }

    /// c(e, f).
    #[inline]
    pub fn get(&self, e: u32, f: u32) -> u32 {
        if e == NULL {
            self.null_row[f as usize]
        } else {
            self.rows[e as usize].get(f)
        }
    }

    /// `out[k · stride] = c(e, fs[k])` for every k: one row's lookups one after the other, with
    /// the row's layout decided once.
    #[inline]
    pub fn gather(&self, e: u32, fs: &[u32], out: &mut [u32], stride: usize) {
        debug_assert!(fs.is_empty() || out.len() > (fs.len() - 1) * stride);
        if e == NULL {
            for (k, &f) in fs.iter().enumerate() {
                out[k * stride] = self.null_row[f as usize];
            }
            return;
        }
        match &self.rows[e as usize] {
            Row::Inline { n, keys, vals } => {
                if *n == 0 {
                    for k in 0..fs.len() {
                        out[k * stride] = 0;
                    }
                    return;
                }
                for (k, &f) in fs.iter().enumerate() {
                    // all INLINE keys compared without a data-dependent loop (unused keys are
                    // EMPTY, never a target id; their value is 0)
                    let mut v = 0;
                    for q in 0..INLINE {
                        v |= if keys[q] == f { vals[q] } else { 0 };
                    }
                    out[k * stride] = v;
                }
            }
            row @ Row::Table { .. } => {
                for (k, &f) in fs.iter().enumerate() {
                    out[k * stride] = row.get(f);
                }
            }
        }
    }

    /// n(e) = Σ_f c(e, f).
    #[inline]
    pub fn total(&self, e: u32) -> u32 {
        self.totals[e as usize]
    }

    /// All row totals, indexed by source id.
    pub fn totals(&self) -> &[u32] {
        &self.totals
    }

    #[inline]
    pub fn inc(&mut self, e: u32, f: u32) {
        self.add(e, f, 1);
    }

    /// Decrement c(e, f); panics if it is 0.
    #[inline]
    pub fn dec(&mut self, e: u32, f: u32) {
        self.add(e, f, -1);
    }

    /// c(e, f) += delta (and n(e)); panics if a count would become negative.
    pub fn add(&mut self, e: u32, f: u32, delta: i32) {
        let d = i64::from(delta);
        let t = &mut self.totals[e as usize];
        *t = check_count(i64::from(*t) + d, f);
        if e == NULL {
            let c = &mut self.null_row[f as usize];
            let old = *c;
            *c = check_count(i64::from(old) + d, f);
            match (old == 0, *c == 0) {
                (true, false) => self.null_entries += 1,
                (false, true) => self.null_entries -= 1,
                _ => {}
            }
        } else {
            debug_assert!((f as usize) < self.null_row.len());
            let change = self.rows[e as usize].add(f, d);
            self.entries = self.entries.wrapping_add_signed(change);
        }
    }

    /// Number of non-zero counts c(e, f), NULL row included.
    pub fn n_entries(&self) -> usize {
        self.entries + self.null_entries
    }

    /// Number of entries in row e.
    pub fn row_len(&self, e: u32) -> usize {
        if e == NULL {
            self.null_entries
        } else {
            self.rows[e as usize].len()
        }
    }

    /// The non-zero counts of row e as (f, c), sorted by f.
    pub fn row_sorted(&self, e: u32) -> Vec<(u32, u32)> {
        let mut v = Vec::with_capacity(self.row_len(e));
        if e == NULL {
            v.extend(
                self.null_row
                    .iter()
                    .enumerate()
                    .filter(|(_, &c)| c != 0)
                    .map(|(f, &c)| (f as u32, c)),
            );
        } else {
            self.rows[e as usize].for_each(|f, c| v.push((f, c)));
            v.sort_unstable();
        }
        v
    }

    /// Whether `self` and `other` hold the same counts (independent of table layout).
    pub fn same_counts(&self, other: &LexCounts) -> bool {
        self.rows.len() == other.rows.len()
            && self.totals == other.totals
            && self.null_row == other.null_row
            && self.entries == other.entries
            && (1..self.rows.len() as u32).all(|e| {
                let a = &self.rows[e as usize];
                a.len() == other.rows[e as usize].len() && {
                    let mut ok = true;
                    a.for_each(|f, c| ok &= other.get(e, f) == c);
                    ok
                }
            })
    }

    /// Rows stored as hash tables (more than [`INLINE`] entries).
    pub fn n_table_rows(&self) -> usize {
        self.rows
            .iter()
            .filter(|r| matches!(r, Row::Table { .. }))
            .count()
    }

    /// Heap bytes used: row headers, tables, totals, the dense NULL row.
    pub fn memory_bytes(&self) -> usize {
        self.rows.capacity() * std::mem::size_of::<Row>()
            + self.rows.iter().map(Row::heap_bytes).sum::<usize>()
            + self.totals.capacity() * 4
            + self.null_row.capacity() * 4
    }
}

/// The distinct words of one sentence pair, for gathering its lexical counts once per sentence
/// ([`LexCounts::gather`]) into a matrix with one column per distinct target word and one entry
/// per distinct source word plus NULL: cell `col(dj)[di]` holds c(e_di, f_dj). Positions map to
/// the distinct indices through `sidx` (source positions 0 … I−1, then I = NULL) and `tidx`.
/// No vocabulary-sized arrays: the ids are sorted and deduplicated per sentence.
#[derive(Clone, Debug, Default)]
pub struct SentenceGather {
    /// Distinct index of each source position; `sidx[I]` is NULL's (the last one).
    pub sidx: Vec<u32>,
    /// Distinct index of each target position.
    pub tidx: Vec<u32>,
    /// Distinct source ids (NULL last).
    pub src_ids: Vec<u32>,
    /// Distinct target ids.
    pub tgt_ids: Vec<u32>,
    keys: Vec<u64>,
}

/// Largest matrix (cells) a sentence is gathered into; longer pairs use direct lookups.
pub const GATHER_MAX_CELLS: usize = 1 << 16;

fn distinct(ids: &[u32], keys: &mut Vec<u64>, idx: &mut Vec<u32>, out: &mut Vec<u32>) {
    keys.clear();
    keys.extend(
        ids.iter()
            .enumerate()
            .map(|(p, &x)| (u64::from(x) << 32) | p as u64),
    );
    keys.sort_unstable();
    idx.clear();
    idx.resize(ids.len(), 0);
    out.clear();
    for &k in keys.iter() {
        let id = (k >> 32) as u32;
        if out.last() != Some(&id) {
            out.push(id);
        }
        idx[k as u32 as usize] = (out.len() - 1) as u32;
    }
}

impl SentenceGather {
    /// Index the sentence pair; returns false (and indexes nothing useful) if its matrix would
    /// have more than [`GATHER_MAX_CELLS`] cells.
    pub fn index(&mut self, src: &[u32], tgt: &[u32]) -> bool {
        distinct(src, &mut self.keys, &mut self.sidx, &mut self.src_ids);
        self.sidx.push(self.src_ids.len() as u32);
        self.src_ids.push(NULL);
        distinct(tgt, &mut self.keys, &mut self.tidx, &mut self.tgt_ids);
        self.rows() * self.tgt_ids.len() <= GATHER_MAX_CELLS
    }

    /// Entries per column (distinct source words + NULL).
    #[inline]
    pub fn rows(&self) -> usize {
        self.src_ids.len()
    }

    /// Fill `mat` (`rows() × tgt_ids.len()`, column-major by target) from `lex`.
    pub fn fill(&self, lex: &LexCounts, mat: &mut Vec<u32>) {
        let rows = self.rows();
        mat.clear();
        mat.resize(rows * self.tgt_ids.len(), 0);
        for (di, &e) in self.src_ids.iter().enumerate() {
            lex.gather(e, &self.tgt_ids, &mut mat[di..], rows);
        }
    }
}

/// Jump counts d(m) over the clamped buckets and their total D.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JumpCounts {
    d: [u32; JUMP_BUCKETS],
    total: u32,
}

impl Default for JumpCounts {
    fn default() -> Self {
        JumpCounts {
            d: [0; JUMP_BUCKETS],
            total: 0,
        }
    }
}

impl JumpCounts {
    /// Bucket index of jump m (clamped to ±MAX_JUMP).
    #[inline]
    pub fn bucket(m: i32) -> usize {
        (m.clamp(-MAX_JUMP, MAX_JUMP) + MAX_JUMP) as usize
    }

    /// d(m), m clamped.
    #[inline]
    pub fn get(&self, m: i32) -> u32 {
        self.d[Self::bucket(m)]
    }

    #[inline]
    pub fn total(&self) -> u32 {
        self.total
    }

    /// Counts by bucket index (bucket k is jump k − MAX_JUMP).
    pub fn buckets(&self) -> &[u32; JUMP_BUCKETS] {
        &self.d
    }

    #[inline]
    pub fn inc(&mut self, m: i32) {
        self.add(m, 1);
    }

    #[inline]
    pub fn dec(&mut self, m: i32) {
        self.add(m, -1);
    }

    #[inline]
    pub fn add(&mut self, m: i32, delta: i32) {
        let b = &mut self.d[Self::bucket(m)];
        *b = check_count(i64::from(*b) + i64::from(delta), m as u32);
        self.total = check_count(i64::from(self.total) + i64::from(delta), m as u32);
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

/// Fertility counts c″(e, n): source tokens of type e with fertility n (clamped to F_MAX).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FertCounts {
    c: Vec<u32>,
}

impl FertCounts {
    pub fn new(n_source: usize) -> Self {
        FertCounts {
            c: vec![0; n_source * FERT_CATS],
        }
    }

    #[inline]
    fn idx(e: u32, n: usize) -> usize {
        e as usize * FERT_CATS + n.min(F_MAX)
    }

    #[inline]
    pub fn get(&self, e: u32, n: usize) -> u32 {
        self.c[Self::idx(e, n)]
    }

    /// c″(e, 0 … F_MAX).
    #[inline]
    pub fn row(&self, e: u32) -> &[u32] {
        let s = e as usize * FERT_CATS;
        &self.c[s..s + FERT_CATS]
    }

    #[inline]
    pub fn inc(&mut self, e: u32, n: usize) {
        self.add(e, n, 1);
    }

    #[inline]
    pub fn dec(&mut self, e: u32, n: usize) {
        self.add(e, n, -1);
    }

    #[inline]
    pub fn add(&mut self, e: u32, n: usize, delta: i32) {
        let c = &mut self.c[Self::idx(e, n)];
        *c = check_count(i64::from(*c) + i64::from(delta), e);
    }

    pub fn clear(&mut self) {
        self.c.fill(0);
    }

    pub fn memory_bytes(&self) -> usize {
        self.c.capacity() * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{Rng, Stream};
    use rustc_hash::FxHashMap;

    /// Naive reference: a HashMap of non-zero counts.
    #[derive(Default)]
    struct Naive {
        c: FxHashMap<(u32, u32), u32>,
        totals: FxHashMap<u32, u32>,
    }

    impl Naive {
        fn add(&mut self, e: u32, f: u32, d: i32) {
            let v = self.c.entry((e, f)).or_insert(0);
            *v = (i64::from(*v) + i64::from(d)) as u32;
            if *v == 0 {
                self.c.remove(&(e, f));
            }
            *self.totals.entry(e).or_insert(0) =
                (i64::from(*self.totals.get(&e).unwrap_or(&0)) + i64::from(d)) as u32;
        }
        fn get(&self, e: u32, f: u32) -> u32 {
            *self.c.get(&(e, f)).unwrap_or(&0)
        }
    }

    fn check(t: &LexCounts, n: &Naive, ns: u32, nt: u32) {
        assert_eq!(t.n_entries(), n.c.len());
        for e in 0..ns {
            assert_eq!(t.total(e), *n.totals.get(&e).unwrap_or(&0), "total {e}");
            let row = t.row_sorted(e);
            let mut want: Vec<(u32, u32)> =
                n.c.iter()
                    .filter(|((a, _), _)| *a == e)
                    .map(|(&(_, f), &c)| (f, c))
                    .collect();
            want.sort_unstable();
            assert_eq!(row, want, "row {e}");
            for f in 0..nt {
                assert_eq!(t.get(e, f), n.get(e, f), "c({e}, {f})");
            }
            // the batched lookup, with a stride
            let fs: Vec<u32> = (0..nt).collect();
            let mut out = vec![u32::MAX; 2 * nt as usize];
            t.gather(e, &fs, &mut out, 2);
            for f in 0..nt {
                assert_eq!(out[2 * f as usize], n.get(e, f), "gather c({e}, {f})");
            }
        }
    }

    #[test]
    fn sentence_gather_matches_lookups() {
        let mut t = LexCounts::new(30, 50);
        let r = Rng::new(5, Stream::Align);
        let mut d = r.draws(0, 0, 0);
        for _ in 0..5000 {
            t.inc(d.below(30), d.below(50));
        }
        let mut g = SentenceGather::default();
        let mut mat = Vec::new();
        for len in [1usize, 2, 7, 40] {
            let src: Vec<u32> = (0..len).map(|_| 1 + d.below(29)).collect();
            let tgt: Vec<u32> = (0..len + 3).map(|_| d.below(50)).collect();
            assert!(g.index(&src, &tgt));
            assert_eq!(g.sidx.len(), src.len() + 1);
            assert_eq!(g.src_ids[g.sidx[src.len()] as usize], NULL);
            g.fill(&t, &mut mat);
            let rows = g.rows();
            for (j, &f) in tgt.iter().enumerate() {
                let col = &mat[g.tidx[j] as usize * rows..][..rows];
                assert_eq!(g.tgt_ids[g.tidx[j] as usize], f);
                for (i, &e) in src.iter().chain([NULL].iter()).enumerate() {
                    assert_eq!(col[g.sidx[i] as usize], t.get(e, f), "{len} {i} {j}");
                }
            }
        }
        // a pair whose distinct words exceed the cap is refused
        let src: Vec<u32> = (1..400).collect();
        let tgt: Vec<u32> = (0..400).collect();
        assert!(!g.index(&src, &tgt));
        // repeats shrink the matrix below the cap
        let src = vec![3u32; 400];
        assert!(g.index(&src, &tgt));
        assert_eq!(g.rows(), 2);
    }

    /// Random inc/dec sequences: rows of very different sizes so that rows go inline → table →
    /// grow → shrink → inline, compared with the naive map after every phase.
    #[test]
    fn random_updates_match_naive() {
        let (ns, nt) = (40u32, 3000u32);
        for seed in 0..4 {
            let r = Rng::new(seed, Stream::Align);
            let mut t = LexCounts::new(ns as usize, nt as usize);
            let mut n = Naive::default();
            let mut live: Vec<(u32, u32)> = Vec::new(); // multiset of current (e, f) tokens
            let mut max_tables = 0;
            for phase in 0..6u32 {
                let mut d = r.draws(0, phase, 0);
                let grow = phase % 2 == 0;
                let steps = 20_000;
                for _ in 0..steps {
                    let ins = live.is_empty() || (d.below(100) < if grow { 70 } else { 30 });
                    if ins {
                        // Row e gets a target range that grows with e: row 0 (NULL) and rows
                        // with few distinct targets stay inline, others become tables.
                        let e = d.below(ns);
                        let span = 1 + (e * e * 2).min(nt - 1);
                        let f = d.below(span);
                        t.inc(e, f);
                        n.add(e, f, 1);
                        live.push((e, f));
                    } else {
                        let k = d.below(live.len() as u32) as usize;
                        let (e, f) = live.swap_remove(k);
                        t.dec(e, f);
                        n.add(e, f, -1);
                    }
                }
                check(&t, &n, ns, nt);
                max_tables = max_tables.max(t.n_table_rows());
            }
            assert!(max_tables > 10, "{max_tables}");
            // Drain completely: every row returns to empty and inline.
            while let Some((e, f)) = live.pop() {
                t.dec(e, f);
                n.add(e, f, -1);
            }
            check(&t, &n, ns, nt);
            assert_eq!(t.n_entries(), 0);
            assert_eq!(t.n_table_rows(), 0);
        }
    }

    #[test]
    fn add_with_larger_deltas() {
        let mut t = LexCounts::new(3, 100);
        let mut n = Naive::default();
        let r = Rng::new(9, Stream::Align);
        let mut d = r.draws(0, 0, 0);
        for _ in 0..5000 {
            let (e, f) = (d.below(3), d.below(100));
            let cur = n.get(e, f) as i32;
            let delta = d.below(7) as i32 - 3;
            let delta = delta.max(-cur);
            t.add(e, f, delta);
            n.add(e, f, delta);
        }
        check(&t, &n, 3, 100);
    }

    #[test]
    fn layout_independent_equality() {
        // Same multiset of updates in two orders: same counts, possibly different layouts.
        let pairs: Vec<(u32, u32)> = (0..2000).map(|k| (1 + k % 3, (k * 7919) % 500)).collect();
        let mut a = LexCounts::new(4, 500);
        let mut b = LexCounts::new(4, 500);
        for &(e, f) in &pairs {
            a.inc(e, f);
        }
        for &(e, f) in pairs.iter().rev() {
            b.inc(e, f);
        }
        assert!(a.same_counts(&b));
        b.dec(pairs[0].0, pairs[0].1);
        assert!(!a.same_counts(&b));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn negative_count_panics() {
        let mut t = LexCounts::new(2, 10);
        t.inc(1, 3);
        t.dec(1, 4);
    }

    #[test]
    fn null_row_is_dense() {
        let mut t = LexCounts::new(2, 10);
        t.inc(NULL, 9);
        t.inc(NULL, 9);
        t.inc(NULL, 0);
        assert_eq!(t.get(NULL, 9), 2);
        assert_eq!(t.total(NULL), 3);
        assert_eq!(t.row_len(NULL), 2);
        assert_eq!(t.row_sorted(NULL), vec![(0, 1), (9, 2)]);
        t.dec(NULL, 0);
        assert_eq!(t.n_entries(), 1);
    }

    #[test]
    fn removal_keeps_probe_chains_intact() {
        // Many keys in one table row, removed in a scattered order: every remaining key must
        // stay reachable after each backward-shift deletion.
        let mut t = LexCounts::new(2, 1 << 16);
        let keys: Vec<u32> = (0..3000u32)
            .map(|k| k.wrapping_mul(2_654_435_761) >> 16)
            .collect();
        let mut uniq = keys.clone();
        uniq.sort_unstable();
        uniq.dedup();
        for &k in &uniq {
            t.inc(1, k);
        }
        for (step, &k) in uniq.iter().enumerate().filter(|(i, _)| i % 3 != 1) {
            t.dec(1, k);
            if step % 97 == 0 {
                for &m in &uniq {
                    let removed = uniq.iter().position(|&x| x == m).unwrap();
                    let gone = removed % 3 != 1 && removed <= step;
                    assert_eq!(t.get(1, m), u32::from(!gone), "key {m} after step {step}");
                }
            }
        }
    }

    #[test]
    fn jump_counts_clamp() {
        let mut j = JumpCounts::default();
        j.inc(1);
        j.inc(200);
        j.inc(64);
        j.inc(-1000);
        assert_eq!(j.get(1), 1);
        assert_eq!(j.get(64), 2);
        assert_eq!(j.get(65), 2);
        assert_eq!(j.get(-64), 1);
        assert_eq!(j.total(), 4);
        j.dec(500);
        assert_eq!(j.get(64), 1);
        assert_eq!(j.total(), 3);
        assert_eq!(JumpCounts::bucket(-64), 0);
        assert_eq!(JumpCounts::bucket(0), 64);
        assert_eq!(JumpCounts::bucket(64), JUMP_BUCKETS - 1);
    }

    #[test]
    fn fert_counts() {
        let mut c = FertCounts::new(3);
        c.inc(2, 0);
        c.inc(2, 40);
        c.inc(2, F_MAX);
        assert_eq!(c.get(2, F_MAX), 2);
        assert_eq!(c.row(2)[0], 1);
        assert_eq!(c.row(1).iter().sum::<u32>(), 0);
        c.dec(2, 33);
        assert_eq!(c.get(2, F_MAX), 1);
    }
}
