//! Click-to-expand virtualized list. Each row collapses by default;
//! clicking expands that row to ~3x its height with a smooth tween,
//! pushing later rows down. Clicking the expanded row again collapses
//! it back. Clicking another row snaps the previous one closed and
//! tweens the new one open.
//!
//! Demonstrates:
//!   - per-row heights via `tree.set_lazy_list_row_height`
//!   - a per-frame hook driving the height from an animated `Signal<f32>`
//!   - `EventCtx::timeline` from inside an `on_click` handler
//!
//! Run with:
//!     cargo run --example lazy_list_expand

mod common;

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

use frostify_gfx::{App, BarSide, Curve, Justify, Len, NodeId, Scene, Signal};

use common::image::hsv;

const W: u32 = 540;
const H: u32 = 600;
const ROWS: u32 = 1000;
const COLLAPSED_H: f32 = 44.0;
const EXPANDED_H: f32 = 140.0;
const ANIM_DUR: Duration = Duration::from_millis(220);
/// Tween key reserved for our height animation. The library's bind
/// system uses 0xC000_0000 + offsets; anything outside that range is
/// safe.
const ANIM_KEY: u32 = 0xE000_0001;

/// Animation state shared between click handlers and the per-frame
/// hook. While `Some`, the per-frame hook pushes the signal's current
/// value into the lazy list's row-height table.
struct AnimSlot {
    row: u32,
    height_sig: Signal<f32>,
}

/// Locate the lazy-list node in the tree. There's only one in this
/// demo; for multi-list scenes the caller would store its id at
/// scene-build time instead of walking.
fn find_list(ctx: &frostify_gfx::SceneCtx) -> Option<NodeId> {
    ctx.tree.iter_ids().find(|id| {
        ctx.tree
            .get(*id)
            .map(|n| n.lazy_list.is_some())
            .unwrap_or(false)
    })
}

fn build(
    s: &mut Scene,
    anim_state: Rc<RefCell<Option<AnimSlot>>>,
    expanded: Rc<std::cell::Cell<Option<u32>>>,
) {
    s.col("root")
        .fill()
        .rgba(0.06, 0.07, 0.09, 1.0)
        .pad(12.0)
        .gap(8.0)
        .child(|p| {
            p.text(
                "header",
                "click any row to expand it — variable heights, no allocation per row",
                14.0,
            )
            .color([1.0, 1.0, 1.0, 0.85]);

            p.lazy_list("playlist", ROWS, COLLAPSED_H, move |row, i| {
                let c = hsv((i as f32 * 0.037).fract(), 0.45, 0.85);
                let anim_state = anim_state.clone();
                let expanded = expanded.clone();
                row.col(format!("row{i}"))
                    .w(Len::Fill)
                    .pad_xy(12.0, 6.0)
                    .gap(4.0)
                    .justify(Justify::Start)
                    .color(c)
                    .radius(6.0)
                    .on_click(move |ctx| {
                        let list_id = match find_list_via_ctx(ctx.tree) {
                            Some(id) => id,
                            None => return,
                        };
                        let was_expanded = expanded.get() == Some(i);

                        // Snap any other expanded row back to default
                        // before starting a new animation.
                        if let Some(prev_row) = expanded.get()
                            && prev_row != i {
                                ctx.tree.set_lazy_list_row_height(
                                    list_id,
                                    prev_row,
                                    COLLAPSED_H,
                                );
                            }

                        let (initial, target) = if was_expanded {
                            (EXPANDED_H, COLLAPSED_H)
                        } else {
                            (COLLAPSED_H, EXPANDED_H)
                        };
                        let height_sig = Signal::new(initial);
                        ctx.timeline.start(
                            ANIM_KEY,
                            height_sig.clone(),
                            target,
                            Curve::EaseInOut,
                            ANIM_DUR,
                            ctx.now,
                        );
                        *anim_state.borrow_mut() = Some(AnimSlot {
                            row: i,
                            height_sig,
                        });
                        expanded.set(if was_expanded { None } else { Some(i) });
                    })
                    .child(|r| {
                        r.text(
                            format!("row_title_{i}"),
                            format!("Track #{i:04}"),
                            14.0,
                        )
                        .color([0.0, 0.0, 0.0, 0.9]);
                        r.text(
                            format!("row_detail_{i}"),
                            "click to expand · album · 3:42",
                            12.0,
                        )
                        .color([0.0, 0.0, 0.0, 0.6]);
                    });
            })
            .w(Len::Fill)
            .h(Len::Fill)
            .scrollbar(|sb| {
                sb.thickness(10.0)
                    .min_thumb(30.0)
                    .margin(4.0)
                    .radius(5.0)
                    .y_side(BarSide::End)
                    .always_visible(true)
                    .thumb_color([0.40, 0.65, 1.00, 0.55])
                    .thumb_hover_color([0.55, 0.80, 1.00, 0.85])
                    .thumb_active_color([0.85, 0.95, 1.00, 1.00])
            });
        });
}

/// `EventCtx` only exposes `&mut NodeTree`, not the wider `SceneCtx`.
/// Walking the tree for the unique lazy-list works fine.
fn find_list_via_ctx(tree: &frostify_gfx::NodeTree) -> Option<NodeId> {
    tree.iter_ids().find(|id| {
        tree.get(*id)
            .map(|n| n.lazy_list.is_some())
            .unwrap_or(false)
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_hal=warn,wgpu_core=warn"),
    )
    .init();

    let anim_state: Rc<RefCell<Option<AnimSlot>>> = Rc::new(RefCell::new(None));
    let expanded: Rc<std::cell::Cell<Option<u32>>> = Rc::new(std::cell::Cell::new(None));

    let scene_anim = anim_state.clone();
    let scene_expanded = expanded.clone();

    let frame_anim = anim_state.clone();

    let app = App::new("lazy_list_expand", W, H)
        .scene(move |s| build(s, scene_anim.clone(), scene_expanded.clone()))
        .on_frame(move |ctx, _timeline, _now| {
            // Push the currently-animating height into the list.
            // No-op when the slot is absent. The set_* call is
            // idempotent on no-change so it's safe to fire every
            // frame; once the tween settles, the signal stops
            // moving and the no-op path returns false.
            let state = frame_anim.borrow();
            let Some(slot) = state.as_ref() else { return };
            let Some(list_id) = find_list(ctx) else { return };
            ctx.tree
                .set_lazy_list_row_height(list_id, slot.row, slot.height_sig.get());
        });

    app.run()
}
