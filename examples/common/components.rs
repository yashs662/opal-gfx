//! Reusable scene components for the frostify-gfx examples.
//!
//! Component shape:
//!   - Each is a plain `fn name(s: &mut Scene, props: Props)`.
//!   - Props own their data (no `&str` / `&[T]` lifetimes leak in).
//!   - Signals are passed by value (clones are cheap `Rc` bumps).
//!   - Closures captured in builder calls are `'static` — props are
//!     moved into the closures rather than referenced.
//!
//! The library itself stays component-free. These live in
//! `examples/common/` so multiple demos can share them without becoming
//! library API surface.

#![allow(dead_code)]

use std::time::Duration;

use frostify_gfx::{
    animated, deps, Align, Computed, Curve, ImageHandle, Len, Scene, Signal, WindowAction,
};

// --- title-bar dot ---

/// One window-control "dot" (close / minimize / maximize). The hover
/// color is auto-derived by brightening the base — sugar via
/// `NodeBuilderRef::hover_color`.
#[derive(Copy, Clone)]
pub struct DotProps {
    pub color: [f32; 4],
    pub action: WindowAction,
}

pub fn dot(s: &mut Scene, props: DotProps) {
    let hov = [
        (props.color[0] + 0.15_f32).min(1.0),
        (props.color[1] + 0.15_f32).min(1.0),
        (props.color[2] + 0.15_f32).min(1.0),
        props.color[3],
    ];
    s.rect("")
        .size_px(14.0, 14.0)
        .color(props.color)
        .radius(7.0)
        .hover_color(hov)
        .window_action(props.action);
}

// --- title bar ---

pub struct TitleBarProps {
    pub title: String,
    pub dots: Vec<DotProps>,
}

/// macOS-style title bar: drag-move on background, 3 close/min/max dots
/// on the left, then a label. The bar itself is the drag handle — the
/// dots intercept their own clicks via `window_action`.
pub fn title_bar(s: &mut Scene, props: TitleBarProps) {
    s.row("title")
        .w(Len::Fill)
        .h_px(56.0)
        .pad(16.0)
        .gap(10.0)
        .align(Align::Center)
        .rgba(0.13, 0.14, 0.18, 1.0)
        .radius(16.0)
        .border(1.0, [1.0, 1.0, 1.0, 0.06])
        .window_action(WindowAction::DragMove)
        .child(|t| {
            for d in props.dots {
                dot(t, d);
            }
            t.text("title_label", &props.title, 16.0)
                .color([1.0, 1.0, 1.0, 0.95]);
        });
}

// --- hero card ---

#[derive(Clone)]
pub struct HeroProps {
    pub lit: Signal<bool>,
    pub hover: Signal<bool>,
    pub pressed: Signal<bool>,
    pub focused: Signal<bool>,
}

/// Big interactive rect demonstrating reactive color (lit toggles base),
/// animated tween (220 ms ease), and the `.on_click` event handler
/// (toggles `lit` from inside the closure). All five reactive concepts
/// in one place.
pub fn hero(s: &mut Scene, props: HeroProps) {
    let color = Computed::new(
        deps!(props.lit, props.hover, props.pressed),
        |(l, h, p)| {
            let base = if l {
                [0.20, 0.95, 0.55, 1.0]
            } else {
                [0.95, 0.25, 0.55, 1.0]
            };
            let lerped = if h {
                [
                    (base[0] + 0.18_f32).min(1.0),
                    (base[1] + 0.18_f32).min(1.0),
                    (base[2] + 0.18_f32).min(1.0),
                    base[3],
                ]
            } else {
                base
            };
            let pf = if p { 0.55_f32 } else { 1.0 };
            [lerped[0] * pf, lerped[1] * pf, lerped[2] * pf, lerped[3]]
        },
    );
    let click_lit = props.lit.clone();
    s.rect("hero")
        .w_px(380.0)
        .h_px(70.0)
        .color(animated(color, Curve::EaseInOut, Duration::from_millis(220)))
        .radius(20.0)
        .border(2.0, [1.0, 1.0, 1.0, 0.85])
        .shadow([0.0, 0.0], 22.0, [0.95, 0.25, 0.55, 1.0], 0.45)
        .on_hover(props.hover)
        .on_press(props.pressed)
        .on_focus(props.focused)
        .on_click(move |_| {
            click_lit.set(!click_lit.get());
        });
}

// --- sidebar ---

pub struct SidebarProps {
    pub art: ImageHandle,
}

/// Spotify-ish sidebar — image header + 3 menu rows.
pub fn sidebar(s: &mut Scene, props: SidebarProps) {
    s.col("sidebar")
        .w_px(200.0)
        .h(Len::Fill)
        .pad(12.0)
        .gap(8.0)
        .rgba(1.0, 1.0, 1.0, 0.04)
        .radius(14.0)
        .child(|c| {
            c.image("art", props.art).size_px(64.0, 64.0).radius(10.0);
            c.text("s0", "Library", 14.0).color([1.0, 1.0, 1.0, 0.85]);
            c.text("s1", "Playlists", 14.0).color([1.0, 1.0, 1.0, 0.55]);
            c.text("s2", "Recent", 14.0).color([1.0, 1.0, 1.0, 0.55]);
        });
}

// --- tooltip ---

/// Tooltip primitive. Visibility is caller-owned: the caller wires
/// `.on_hover_dwell(delay, move |_| visible.set(true))` and
/// `.on_hover(visible_signal_clone_inverted_via_Computed)` on the anchor
/// node — this component just renders the bubble. Doctrine: no magic.
/// The component does not allocate the visibility signal; the caller
/// does, and can grep `Signal::new(false)` to find every tooltip
/// allocation site.
///
/// Position is logical-px absolute relative to the parent scope. For an
/// anchor at `[ax, ay]` with the tooltip drawn below, pass
/// `pos: [ax, ay + anchor_h + 6.0]`.
pub struct TooltipProps {
    pub visible: Signal<bool>,
    pub text: String,
    pub pos: [f32; 2],
    pub font_size: f32,
}

pub fn tooltip(s: &mut Scene, props: TooltipProps) {
    let v_bg = props.visible.clone();
    let v_text = props.visible;
    let bg = Computed::new(deps!(v_bg), |(v,)| {
        let a = if v { 0.92_f32 } else { 0.0 };
        [0.10, 0.11, 0.14, a]
    });
    let fg = Computed::new(deps!(v_text), |(v,)| {
        let a = if v { 1.0_f32 } else { 0.0 };
        [1.0, 1.0, 1.0, a]
    });
    s.rect("")
        .abs(props.pos[0], props.pos[1])
        .pad_xy(8.0, 4.0)
        .color(animated(bg, Curve::EaseInOut, Duration::from_millis(120)))
        .radius(6.0)
        .border(1.0, [1.0, 1.0, 1.0, 0.06])
        .child(|t| {
            t.text((), &props.text, props.font_size).color(animated(
                fg,
                Curve::EaseInOut,
                Duration::from_millis(120),
            ));
        });
}

// --- slider (on_drag) ---

/// Horizontal slider built on the generic `on_drag` primitive. The
/// track is the drag node; dragging maps the cursor's x to a 0..1
/// fraction (clamped) and writes it into the caller-owned `value`
/// signal. A fill rect mirrors the value via `width_pct`. Caller owns
/// `value` — read it anywhere, or wrap it in a `Computed` for derived
/// state. (Click-to-set without a drag is intentionally not wired here;
/// the first cursor move during a press sets the position.)
#[derive(Clone)]
pub struct SliderProps {
    pub value: Signal<f32>,
    pub width: f32,
}

pub fn slider(s: &mut Scene, props: SliderProps) {
    // `width_pct` / Len::Pct is a 0.0..=1.0 fraction of the parent's
    // content width — NOT a 0..100 percentage.
    let pct = Computed::new(deps!(props.value), |(v,)| v.clamp(0.0, 1.0));
    let drag_value = props.value;
    s.row("slider_track")
        .w_px(props.width)
        .h_px(8.0)
        .radius(4.0)
        .rgba(1.0, 1.0, 1.0, 0.15)
        .align(Align::Center)
        .on_drag(move |d| {
            let r = d.tree.get(d.node).map(|n| n.rect).unwrap_or([0.0; 4]);
            if r[2] > 0.0 {
                let frac = ((d.current[0] - r[0]) / r[2]).clamp(0.0, 1.0);
                drag_value.set(frac);
            }
        })
        .child(|t| {
            t.rect("slider_fill")
                .h(Len::Fill)
                .radius(4.0)
                .rgba(0.20, 0.55, 0.95, 1.0)
                .width_pct(pct);
        });
}

// --- modal dialog ---

/// Modal dialog convention. Render this **only when the modal should be
/// visible** — the caller gates it with `if visible.get() { modal(s, …) }`
/// inside the scene closure and flips `visible` + the rebuild token to
/// show/hide.
///
/// Structure: a full-window scrim (`dismiss_transparent`, so a press on
/// it triggers `App::on_unhandled_press`) that centers an opaque panel.
/// The panel carries an absorbing `on_click` so presses *inside* it hit
/// the panel — not the scrim — and therefore don't dismiss. Wire the
/// dismiss in `main`:
///
/// ```ignore
/// let rebuild = app.rebuild_token();
/// let vis = visible.clone();
/// app = app.on_unhandled_press(move |_| { vis.set(false); rebuild.set(true); });
/// ```
pub struct ModalProps {
    pub title: String,
    pub body: String,
}

pub fn modal(s: &mut Scene, props: ModalProps) {
    s.col("modal_overlay")
        .abs(0.0, 0.0)
        .w(Len::Fill)
        .h(Len::Fill)
        .rgba(0.0, 0.0, 0.0, 0.55)
        .center()
        .dismiss_transparent()
        .child(|o| {
            o.col("modal_panel")
                .w_px(360.0)
                .pad(24.0)
                .gap(12.0)
                .rgba(0.13, 0.14, 0.18, 1.0)
                .radius(16.0)
                .border(1.0, [1.0, 1.0, 1.0, 0.10])
                .shadow([0.0, 8.0], 32.0, [0.0, 0.0, 0.0, 1.0], 0.45)
                // Absorb clicks: a press inside the panel hits this node,
                // not the scrim, so it doesn't count as an outside press.
                .on_click(|_| {})
                .child(|p| {
                    p.text("modal_title", &props.title, 20.0)
                        .color([1.0, 1.0, 1.0, 0.95]);
                    p.text("modal_body", &props.body, 14.0)
                        .color([1.0, 1.0, 1.0, 0.7]);
                });
        });
}

// --- animated blob (rect + label pair) ---

#[derive(Clone)]
pub struct BlobProps {
    pub pos: Signal<[f32; 2]>,
    pub size: Signal<[f32; 2]>,
}

/// Position-and-size tweened rect plus a small label glued to the same
/// position. Emitted as two sibling nodes into the caller's parent —
/// declared in painter-order order, so the rect paints under the label
/// even though `Bind<Position>` keeps them visually together.
pub fn blob(s: &mut Scene, props: BlobProps) {
    let pos_rect = props.pos.clone();
    let pos_label = props.pos.clone();
    s.rect("blob")
        .pos(animated(pos_rect, Curve::EaseInOut, Duration::from_millis(260)))
        .size_bind(animated(
            props.size,
            Curve::EaseInOut,
            Duration::from_millis(260),
        ))
        .rgba(0.10, 0.85, 0.95, 1.0)
        .radius(28.0)
        .border(2.0, [1.0, 1.0, 1.0, 0.85])
        .shadow([0.0, 6.0], 18.0, [0.10, 0.85, 0.95, 1.0], 0.55);
    s.text("blob_lbl", "TOP", 16.0)
        .pos(animated(
            pos_label,
            Curve::EaseInOut,
            Duration::from_millis(260),
        ))
        .color([0.0, 0.0, 0.0, 0.9]);
}
