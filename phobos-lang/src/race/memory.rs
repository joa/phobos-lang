// A CTA's shared memory as the race check sees it: for every byte, who
// wrote it and who read it since the last barrier. Two threads touching one
// byte in one epoch, at least one of them writing, is a race. Threads of one
// warp only race when no warp-wide op ran between their accesses, and an
// asynchronous copy keeps its bytes written, across barriers, until the
// thread that issued it waits for it.
//
// An access whose address depended on data covers everything its view can
// reach. It is kept as one range per op, not byte by byte, and a conflict
// with one is only possible, not certain.

use std::collections::HashMap;

/// One access to one byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Touch {
    pub thread: u32,
    /// The op, an index into the program's ops.
    pub site: u32,
    pub write: bool,
    /// The warp's count of warp-wide ops when the access was made.
    pub stamp: u64,
    /// Made by a warp-wide op on the whole warp's behalf.
    pub collective: bool,
    /// The address was known. Otherwise the access covers its whole view.
    pub exact: bool,
}

#[derive(Clone, Copy, Default)]
struct Cell {
    writer: Option<Touch>,
    reader: Option<Touch>,
    /// A reader from another thread than `reader`, and one from another
    /// warp: with the first, all a write needs to tell whether another
    /// thread, and another warp, read the byte.
    other_reader: Option<Touch>,
    far_reader: Option<Touch>,
    in_flight: bool,
}

impl Cell {
    fn is_empty(&self) -> bool {
        self.writer.is_none() && self.reader.is_none()
    }
}

/// Every unknown-address access one op made into one buffer this epoch.
struct Range {
    lo: i64,
    hi: i64,
    touch: Touch,
    /// Made by more than one thread.
    shared: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Level {
    /// Threads of different warps.
    Cta,
    /// Lanes of one warp with no warp-wide op between them.
    Warp,
    /// One side's address was not known.
    Possible,
}

#[derive(Clone, Debug)]
pub(super) struct Hazard {
    pub level: Level,
    pub first: Touch,
    pub second: Touch,
    pub buffer: usize,
    pub byte: i64,
    pub count: usize,
}

struct InFlight {
    thread: u32,
    buffer: usize,
    lo: i64,
    hi: i64,
    touch: Touch,
    group: Option<u64>,
}

pub(super) struct Memory {
    pub names: Vec<String>,
    sizes: Vec<i64>,
    cells: Vec<Vec<Cell>>,
    touched: Vec<(usize, usize)>,
    /// Per buffer, the bytes exact accesses reached this epoch.
    reached: Vec<Option<(i64, i64)>>,
    ranges: HashMap<(usize, u32, bool), Range>,
    hazards: HashMap<(u32, u32, bool, bool, Level), Hazard>,
    pub out_of_bounds: Vec<(Touch, usize, i64)>,
    in_flight: Vec<InFlight>,
    groups: HashMap<u32, u64>,
    pub epochs: usize,
    pub bytes: u64,
}

impl Memory {
    pub(super) fn new(buffers: Vec<(String, i64)>) -> Memory {
        let (names, sizes): (Vec<_>, Vec<_>) = buffers.into_iter().unzip();
        let cells = sizes.iter().map(|_| Vec::new()).collect();
        let reached = vec![None; sizes.len()];
        Memory {
            names,
            sizes,
            cells,
            touched: Vec::new(),
            reached,
            ranges: HashMap::new(),
            hazards: HashMap::new(),
            out_of_bounds: Vec::new(),
            in_flight: Vec::new(),
            groups: HashMap::new(),
            epochs: 1,
            bytes: 0,
        }
    }

    /// The bytes `lo..hi` of `buffer`, touched once.
    pub(super) fn touch(&mut self, buffer: usize, lo: i64, hi: i64, t: Touch) {
        if t.exact {
            self.mark(buffer, lo, hi, t, false);
        } else {
            self.range(buffer, lo, hi, t);
        }
    }

    fn mark(&mut self, buffer: usize, lo: i64, hi: i64, t: Touch, in_flight: bool) {
        let size = self.sizes[buffer];
        if (lo < 0 || hi > size) && self.out_of_bounds.len() < 16 {
            self.out_of_bounds
                .push((t, buffer, if lo < 0 { lo } else { hi }));
        }
        let (lo, hi) = (lo.max(0), hi.min(size));
        if lo >= hi {
            return;
        }
        if self.cells[buffer].is_empty() {
            self.cells[buffer] = vec![Cell::default(); size as usize];
        }
        self.bytes += (hi - lo) as u64;
        let reached = self.reached[buffer].get_or_insert((lo, hi));
        *reached = (reached.0.min(lo), reached.1.max(hi));
        let ranged: Vec<Touch> = self
            .ranges
            .iter()
            .filter(|((b, _, _), r)| {
                *b == buffer
                    && r.lo < hi
                    && lo < r.hi
                    && (r.touch.write || t.write)
                    && (r.shared || r.touch.thread != t.thread)
            })
            .map(|(_, r)| r.touch)
            .collect();
        for r in ranged {
            self.conflict(r, t, buffer, lo);
        }
        for byte in lo..hi {
            let cell = self.cells[buffer][byte as usize];
            let mut found = Vec::new();
            if t.write {
                found.extend(cell.writer);
                found.extend(cell.reader);
                found.extend(cell.other_reader);
                found.extend(cell.far_reader);
            } else {
                found.extend(cell.writer);
            }
            for earlier in found {
                if earlier.thread != t.thread || (cell.in_flight && !t.write && earlier.write) {
                    self.conflict(earlier, t, buffer, byte);
                }
            }
            if cell.is_empty() {
                self.touched.push((buffer, byte as usize));
            }
            let cell = &mut self.cells[buffer][byte as usize];
            if t.write {
                cell.writer = Some(t);
                cell.in_flight = in_flight;
            } else if let Some(first) = cell.reader {
                if first.thread != t.thread && cell.other_reader.is_none() {
                    cell.other_reader = Some(t);
                }
                if first.thread / 32 != t.thread / 32 && cell.far_reader.is_none() {
                    cell.far_reader = Some(t);
                }
            } else {
                cell.reader = Some(t);
            }
        }
    }

    /// An access whose address was not known: one range per op, checked
    /// against the exact accesses it overlaps and the other ranges.
    fn range(&mut self, buffer: usize, lo: i64, hi: i64, t: Touch) {
        let (lo, hi) = (lo.max(0), hi.min(self.sizes[buffer]));
        if lo >= hi {
            return;
        }
        let key = (buffer, t.site, t.write);
        if let Some(r) = self.ranges.get_mut(&key) {
            r.shared |= r.touch.thread != t.thread;
            if r.lo <= lo && hi <= r.hi {
                return;
            }
            r.lo = r.lo.min(lo);
            r.hi = r.hi.max(hi);
        } else {
            let others: Vec<Touch> = self
                .ranges
                .iter()
                .filter(|((b, _, _), r)| {
                    *b == buffer && r.lo < hi && lo < r.hi && (r.touch.write || t.write)
                })
                .map(|(_, r)| r.touch)
                .collect();
            for r in others {
                self.conflict(r, t, buffer, lo);
            }
            self.ranges.insert(
                key,
                Range {
                    lo,
                    hi,
                    touch: t,
                    shared: false,
                },
            );
        }
        // Against the exact accesses, only where they reached this epoch.
        let Some((rlo, rhi)) = self.reached[buffer] else {
            return;
        };
        let (lo, hi) = (lo.max(rlo), hi.min(rhi));
        for byte in lo..hi {
            let cell = self.cells[buffer][byte as usize];
            let earlier = if t.write {
                cell.writer.or(cell.reader)
            } else {
                cell.writer
            };
            if let Some(e) = earlier
                && e.thread != t.thread
            {
                self.conflict(e, t, buffer, byte);
                break;
            }
        }
    }

    fn conflict(&mut self, a: Touch, b: Touch, buffer: usize, byte: i64) {
        let same_warp = a.thread / 32 == b.thread / 32 && a.thread != b.thread;
        let level = if !a.exact || !b.exact {
            Level::Possible
        } else if same_warp {
            // A warp-wide op between them, or one they made together,
            // orders the lanes. Two lanes storing through one instruction
            // are one store of the warp, as when a lane pair writes the sum
            // a shuffle gave both.
            let one_store = a.site == b.site && a.write && b.write;
            if a.collective || b.collective || a.stamp != b.stamp || one_store {
                return;
            }
            Level::Warp
        } else {
            Level::Cta
        };
        let key = (a.site, b.site, a.write, b.write, level);
        self.hazards
            .entry(key)
            .and_modify(|h| h.count += 1)
            .or_insert(Hazard {
                level,
                first: a,
                second: b,
                buffer,
                byte,
                count: 1,
            });
    }

    /// Every thread passed a barrier: start a new epoch, with the copies
    /// still in flight written again by their threads.
    pub(super) fn barrier(&mut self) {
        for (buffer, byte) in self.touched.drain(..) {
            self.cells[buffer][byte] = Cell::default();
        }
        self.reached.iter_mut().for_each(|r| *r = None);
        self.ranges.clear();
        self.epochs += 1;
        let pending: Vec<_> = self
            .in_flight
            .iter()
            .map(|f| (f.buffer, f.lo, f.hi, f.touch))
            .collect();
        for (buffer, lo, hi, touch) in pending {
            self.mark(buffer, lo, hi, touch, true);
        }
    }

    /// An asynchronous copy into `lo..hi`, written until its thread waits
    /// for its group.
    pub(super) fn copy_async(&mut self, buffer: usize, lo: i64, hi: i64, t: Touch) {
        self.mark(buffer, lo, hi, t, true);
        self.in_flight.push(InFlight {
            thread: t.thread,
            buffer,
            lo,
            hi,
            touch: t,
            group: None,
        });
    }

    /// Closes a thread's open copies into one group.
    pub(super) fn commit(&mut self, thread: u32) {
        let next = self.groups.entry(thread).or_insert(0);
        *next += 1;
        let id = *next;
        for f in self
            .in_flight
            .iter_mut()
            .filter(|f| f.thread == thread && f.group.is_none())
        {
            f.group = Some(id);
        }
    }

    /// A thread waits until at most `pending` of its groups are in flight.
    pub(super) fn wait(&mut self, thread: u32, pending: i64) {
        let newest = self.groups.get(&thread).copied().unwrap_or(0);
        let keep_from = newest.saturating_sub(pending.max(0) as u64) + 1;
        let (done, kept): (Vec<_>, Vec<_>) = self
            .in_flight
            .drain(..)
            .partition(|f| f.thread == thread && f.group.is_some_and(|g| g < keep_from));
        self.in_flight = kept;
        for f in done {
            for byte in f.lo.max(0)..f.hi.min(self.sizes[f.buffer]) {
                if let Some(cell) = self.cells[f.buffer].get_mut(byte as usize) {
                    cell.in_flight = false;
                }
            }
        }
    }

    pub(super) fn hazards(&self) -> Vec<Hazard> {
        let mut out: Vec<_> = self.hazards.values().cloned().collect();
        out.sort_by_key(|h| (h.first.site, h.second.site));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(thread: u32, site: u32, write: bool) -> Touch {
        Touch {
            thread,
            site,
            write,
            stamp: 0,
            collective: false,
            exact: true,
        }
    }

    fn levels(m: &Memory) -> Vec<Level> {
        m.hazards().iter().map(|h| h.level).collect()
    }

    #[test]
    fn another_warps_read_of_a_write_races_until_a_barrier() {
        let mut m = Memory::new(vec![("s".into(), 64)]);
        m.touch(0, 0, 4, touch(0, 1, true));
        m.touch(0, 0, 4, touch(40, 2, false));
        assert_eq!(levels(&m), vec![Level::Cta]);

        let mut m = Memory::new(vec![("s".into(), 64)]);
        m.touch(0, 0, 4, touch(0, 1, true));
        m.barrier();
        m.touch(0, 0, 4, touch(40, 2, false));
        assert!(levels(&m).is_empty());
    }

    #[test]
    fn a_threads_own_bytes_and_shared_reads_do_not_race() {
        let mut m = Memory::new(vec![("s".into(), 64)]);
        m.touch(0, 0, 4, touch(3, 1, true));
        m.touch(0, 0, 4, touch(3, 2, false));
        m.touch(0, 8, 12, touch(1, 3, false));
        m.touch(0, 8, 12, touch(70, 4, false));
        assert!(levels(&m).is_empty());
    }

    #[test]
    fn a_write_after_another_threads_read_races() {
        let mut m = Memory::new(vec![("s".into(), 64)]);
        m.touch(0, 0, 4, touch(1, 1, false));
        m.touch(0, 0, 4, touch(2, 1, false));
        m.touch(0, 0, 4, touch(1, 2, true));
        assert_eq!(levels(&m), vec![Level::Warp]);
    }

    /// Two lanes of warp 0 read first; a third reader in warp 1 still makes
    /// lane 0's write a race between warps.
    #[test]
    fn a_reader_in_another_warp_is_remembered() {
        let mut m = Memory::new(vec![("s".into(), 64)]);
        m.touch(0, 0, 4, touch(0, 1, false));
        m.touch(0, 0, 4, touch(1, 1, false));
        m.touch(0, 0, 4, touch(33, 1, false));
        m.touch(0, 0, 4, touch(0, 2, true));
        assert!(levels(&m).contains(&Level::Cta), "{:?}", levels(&m));
    }

    #[test]
    fn two_lanes_storing_through_one_instruction_are_one_store() {
        let mut m = Memory::new(vec![("s".into(), 64)]);
        m.touch(0, 0, 4, touch(0, 1, true));
        m.touch(0, 0, 4, touch(1, 1, true));
        assert!(levels(&m).is_empty());
        // From two warps, it is a race.
        m.touch(0, 0, 4, touch(40, 1, true));
        assert_eq!(levels(&m), vec![Level::Cta]);
    }

    #[test]
    fn a_copy_in_flight_outlives_a_barrier_until_waited_for() {
        let mut m = Memory::new(vec![("s".into(), 64)]);
        m.copy_async(0, 0, 16, touch(0, 1, true));
        m.commit(0);
        m.barrier();
        m.touch(0, 0, 4, touch(64, 2, false));
        assert_eq!(levels(&m), vec![Level::Cta]);

        let mut m = Memory::new(vec![("s".into(), 64)]);
        m.copy_async(0, 0, 16, touch(0, 1, true));
        m.commit(0);
        m.wait(0, 0);
        m.barrier();
        m.touch(0, 0, 4, touch(64, 2, false));
        assert!(levels(&m).is_empty());
    }

    #[test]
    fn a_gather_from_a_table_is_only_possibly_racing() {
        let gather = |thread| Touch {
            exact: false,
            ..touch(thread, 5, false)
        };
        // Written last epoch, gathered from this one: nothing to report.
        let mut m = Memory::new(vec![("s".into(), 4096)]);
        m.touch(0, 0, 4096, touch(0, 1, true));
        m.barrier();
        for t in 0..256 {
            m.touch(0, 0, 4096, gather(t));
        }
        assert!(levels(&m).is_empty());
        // Written in the same epoch by another thread: possibly racing.
        m.touch(0, 100, 104, touch(7, 6, true));
        assert_eq!(levels(&m), vec![Level::Possible]);
    }
}
