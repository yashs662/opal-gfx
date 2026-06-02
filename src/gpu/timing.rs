//! GPU timestamp query resources with per-pass split.
//!
//! Queries are assigned dynamically per frame: each pass that runs
//! allocates the next pair `(2i, 2i+1)`, then `resolve_query_set` reads
//! back only the contiguous prefix that was actually written. Unwritten
//! queries are never resolved — avoids undefined-behaviour on backends
//! that don't tolerate gaps.
//!
//! Readback is **asynchronous**: each frame picks an idle readback slot
//! from a 2-buffer ring, kicks `map_async`, and drains completed
//! callbacks via non-blocking `device.poll(Poll)`. The CPU never stalls
//! on the GPU. Reported stats are at most one frame behind.

use std::sync::mpsc::{self, Receiver, Sender};

const SLOTS: usize = 2;
/// Four logical passes: opaque, final, overdraw count (optional),
/// overdraw compose (optional). Max queries per frame = 8.
pub const MAX_PASSES: usize = 4;
pub const QUERY_COUNT: u32 = (MAX_PASSES * 2) as u32;

/// Per-slot readback buffer size. Must be ≥ `MAX_PASSES * 2 * 8` bytes
/// and aligned to `wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT` (256).
const RESOLVE_BYTES: wgpu::BufferAddress = 256;

// Pass IDs — stable handles referenced from `FrameTiming`. The runtime
// index inside the query set is assigned dynamically per frame.
pub const PASS_OPAQUE: usize = 0;
pub const PASS_FINAL: usize = 1;
pub const PASS_OD_COUNT: usize = 2;
pub const PASS_OD_COMPOSE: usize = 3;

/// Assigns query-pair indices to passes as they run within a single
/// frame. `pair_of[pass_id] = Some(n)` means pass `pass_id` owns
/// queries `2n` (begin) and `2n + 1` (end). `None` means the pass did
/// not run this frame.
#[derive(Copy, Clone, Debug, Default)]
pub struct PassAlloc {
    pair_of: [Option<u8>; MAX_PASSES],
    next_pair: u8,
}

impl PassAlloc {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a pair for `pass_id`. Returns `(begin_query_idx,
    /// end_query_idx)` to hand to the pass's timestamp write descriptor.
    /// Panics if called twice for the same pass in one frame.
    pub fn alloc(&mut self, pass_id: usize) -> (u32, u32) {
        debug_assert!(
            self.pair_of[pass_id].is_none(),
            "duplicate alloc for pass {pass_id}"
        );
        let pair = self.next_pair;
        self.pair_of[pass_id] = Some(pair);
        self.next_pair += 1;
        let begin = (pair as u32) * 2;
        (begin, begin + 1)
    }

    /// Total queries written this frame.
    pub fn query_count(&self) -> u32 {
        (self.next_pair as u32) * 2
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct FrameTiming {
    pub total_ms: f32,
    pub opaque_ms: f32,
    pub final_ms: f32,
    pub overdraw_ms: f32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SlotState {
    /// Nothing written to this buffer yet; safe to copy into.
    Idle,
    /// Copy encoded + `map_async` kicked; waiting for callback.
    Pending,
}

struct ReadbackSlot {
    buffer: wgpu::Buffer,
    state: SlotState,
    /// Snapshot of the PassAlloc that produced this slot's data.
    /// Interpreted by `parse_frame_timing` on readback.
    alloc: PassAlloc,
    /// Number of u64s actually written this frame. The map_async
    /// request covers `bytes` bytes; anything past that is ignored.
    bytes: wgpu::BufferAddress,
}

pub struct Timing {
    pub query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    slots: [ReadbackSlot; SLOTS],
    /// Slot that received this frame's copy (set by `encode_resolve`,
    /// consumed by `kick_map_async`). `None` when no idle slot was
    /// available and we skipped the copy.
    pending_kick: Option<usize>,
    period_ns: f32,
    map_tx: Sender<MapMsg>,
    map_rx: Receiver<MapMsg>,
    last: Option<FrameTiming>,
}

type MapMsg = (usize, Result<(), wgpu::BufferAsyncError>);

impl Timing {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("frostify.timing qs"),
            ty: wgpu::QueryType::Timestamp,
            count: QUERY_COUNT,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frostify.timing resolve"),
            size: RESOLVE_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mk_slot_buffer = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: RESOLVE_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let slots = [
            ReadbackSlot {
                buffer: mk_slot_buffer("frostify.timing readback[0]"),
                state: SlotState::Idle,
                alloc: PassAlloc::new(),
                bytes: 0,
            },
            ReadbackSlot {
                buffer: mk_slot_buffer("frostify.timing readback[1]"),
                state: SlotState::Idle,
                alloc: PassAlloc::new(),
                bytes: 0,
            },
        ];
        let period_ns = queue.get_timestamp_period();
        let (map_tx, map_rx) = mpsc::channel();
        Self {
            query_set,
            resolve,
            slots,
            pending_kick: None,
            period_ns,
            map_tx,
            map_rx,
            last: None,
        }
    }

    /// Resolve the contiguous prefix of queries written this frame into
    /// the next idle readback slot. `alloc` describes the
    /// pass→query-pair assignment used when encoding. Must be called
    /// after every pass has been encoded, before `encoder.finish`. If
    /// no slot is idle, the copy is skipped.
    pub fn encode_resolve(&mut self, encoder: &mut wgpu::CommandEncoder, alloc: PassAlloc) {
        let count = alloc.query_count();
        if count == 0 {
            self.pending_kick = None;
            return;
        }
        let idx = match self.slots.iter().position(|s| s.state == SlotState::Idle) {
            Some(i) => i,
            None => {
                self.pending_kick = None;
                return;
            }
        };
        let bytes = (count as u64) * 8;
        encoder.resolve_query_set(&self.query_set, 0..count, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.slots[idx].buffer, 0, bytes);
        self.slots[idx].state = SlotState::Pending;
        self.slots[idx].alloc = alloc;
        self.slots[idx].bytes = bytes;
        self.pending_kick = Some(idx);
    }

    /// Kick `map_async` on the slot that received this frame's copy.
    /// Must be called after `queue.submit` on the encoder containing the
    /// resolve+copy. The callback runs on the wgpu worker and pushes a
    /// message into `map_rx`; `poll` drains it.
    pub fn kick_map_async(&mut self) {
        let Some(idx) = self.pending_kick.take() else {
            return;
        };
        let bytes = self.slots[idx].bytes;
        if bytes == 0 {
            return;
        }
        let tx = self.map_tx.clone();
        self.slots[idx]
            .buffer
            .slice(0..bytes)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send((idx, r));
            });
    }

    /// Non-blocking poll. Advances wgpu's internal mapping machinery and
    /// drains any callbacks that fired since the last call. Updates
    /// `last` for every slot whose map completed this tick; all
    /// finished slots are unmapped and returned to `Idle`.
    pub fn poll(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::Poll);
        while let Ok((idx, res)) = self.map_rx.try_recv() {
            if res.is_err() {
                self.reset_slot(idx);
                continue;
            }
            let alloc = self.slots[idx].alloc;
            let bytes = self.slots[idx].bytes;
            let period = self.period_ns;
            let timing = {
                let view = self.slots[idx].buffer.slice(0..bytes).get_mapped_range();
                let raw: &[u64] = bytemuck::cast_slice(&view);
                parse_frame_timing(raw, &alloc, period)
            };
            self.slots[idx].buffer.unmap();
            self.reset_slot(idx);
            self.last = Some(timing);
        }
    }

    fn reset_slot(&mut self, idx: usize) {
        self.slots[idx].state = SlotState::Idle;
        self.slots[idx].alloc = PassAlloc::new();
        self.slots[idx].bytes = 0;
    }

    pub fn last(&self) -> Option<FrameTiming> {
        self.last
    }
}

fn parse_frame_timing(raw: &[u64], alloc: &PassAlloc, period_ns: f32) -> FrameTiming {
    let ms_for = |pass_id: usize| -> f32 {
        let Some(pair) = alloc.pair_of[pass_id] else {
            return 0.0;
        };
        let b = raw[(pair as usize) * 2];
        let e = raw[(pair as usize) * 2 + 1];
        (e.saturating_sub(b) as f32) * period_ns / 1_000_000.0
    };
    let mut t = FrameTiming {
        opaque_ms: ms_for(PASS_OPAQUE),
        final_ms: ms_for(PASS_FINAL),
        overdraw_ms: ms_for(PASS_OD_COUNT) + ms_for(PASS_OD_COMPOSE),
        total_ms: 0.0,
    };
    t.total_ms = t.opaque_ms + t.final_ms + t.overdraw_ms;
    t
}

/// Aggregate frame stats published by the renderer each frame.
#[derive(Copy, Clone, Debug, Default)]
pub struct FrameStats {
    pub cpu_ms: f32,
    pub gpu_ms: f32,
    pub opaque_ms: f32,
    pub final_ms: f32,
    pub overdraw_ms: f32,
    pub instance_count: u32,
    pub opaque_count: u32,
    pub glass_count: u32,
    pub drawcalls: u32,
    pub dirty_mask: u32,
    // Compositor (P1) — layer-tree accounting. Single root layer today,
    // so `layer_count == 1`, `raster_count`/`composite_count` are 0 or 1.
    /// Live layers in the layer tree.
    pub layer_count: u32,
    /// Layers that re-rasterized their texture this frame.
    pub raster_count: u32,
    /// Layers composited to the surface this frame.
    pub composite_count: u32,
    /// Total bytes the layer textures occupy (projected in P1).
    pub layer_vram: u64,
}
