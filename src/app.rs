use std::path::PathBuf;
use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::debug;
use crate::gpu::{GpuContext, ShapeInstance};
use crate::node::{Node, NodeId, NodeTree};
use crate::signal::Signal;

type WindowArc = Arc<Window>;

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub decorations: bool,
    pub capture_dir: PathBuf,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            title: "frostify-gfx".into(),
            width: 1100,
            height: 750,
            decorations: false,
            capture_dir: PathBuf::from("debug_captures"),
        }
    }
}

struct DemoIds {
    hero: NodeId,
}

pub struct App {
    config: AppConfig,
    window: Option<WindowArc>,
    gpu: Option<GpuContext>,
    tree: NodeTree,
    instances: Vec<ShapeInstance>,
    demo: DemoIds,
    /// Stage-1 demo signal: hero rect alternate color on Space.
    hero_lit: Signal<bool>,
}

impl App {
    pub fn new(config: AppConfig) -> Self {
        let (tree, demo) = build_demo_tree(config.width as f32, config.height as f32);
        log::info!("demo scene: {} nodes", tree.len());
        Self {
            config,
            window: None,
            gpu: None,
            tree,
            instances: Vec::new(),
            demo,
            hero_lit: Signal::new(false),
        }
    }

    pub fn run(config: AppConfig) -> Result<(), Box<dyn std::error::Error>> {
        let event_loop = EventLoop::new()?;
        let mut app = App::new(config);
        event_loop.run_app(&mut app)?;
        Ok(())
    }

    /// If the tree's dirty mask is set, re-flatten and push the new
    /// instance list to the GPU. Returns true if anything was uploaded.
    fn flush_tree(&mut self) -> bool {
        if self.tree.take_dirty() == 0 {
            return false;
        }
        self.instances = self.tree.flatten();
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_instances(&self.instances);
        }
        true
    }

    fn autocapture(&mut self) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        let (rgba, w, h) = gpu.capture_rgba();
        let path = debug::screenshot_path(&self.config.capture_dir);
        match debug::save_png(&path, &rgba, w, h) {
            Ok(()) => log::info!("auto-capture saved: {}", path.display()),
            Err(e) => log::error!("auto-capture failed: {e}"),
        }
    }
}

/// Stage-1 demo scene exercising the retained tree:
///   - dark window panel (root) with shadow
///   - title bar (child)
///   - 8×5 swatch grid (children of a row container)
///   - hero accent rect with border (tracked for Space-key recolor)
///
/// All children inherit their parent's absolute position via `flatten()`.
fn build_demo_tree(w: f32, h: f32) -> (NodeTree, DemoIds) {
    let mut tree = NodeTree::new();

    let panel_pad = 24.0;
    let panel = tree.add_root(
        Node::rect()
            .pos(panel_pad, panel_pad)
            .size(w - panel_pad * 2.0, h - panel_pad * 2.0)
            .rgba(0.06, 0.07, 0.10, 0.92)
            .radius(28.0)
            .border(1.5, [1.0, 1.0, 1.0, 0.10])
            .shadow([0.0, 16.0], 40.0, [0.0, 0.0, 0.0, 1.0], 0.55)
            .build(),
    );

    // Title bar.
    tree.add_child(
        panel,
        Node::rect()
            .pos(20.0, 20.0)
            .size(w - panel_pad * 2.0 - 40.0, 56.0)
            .rgba(0.13, 0.14, 0.18, 1.0)
            .radius(16.0)
            .border(1.0, [1.0, 1.0, 1.0, 0.06])
            .build(),
    );

    // Three traffic-light style dots inside the title bar.
    let dot_y = 40.0;
    let dot_colors = [
        [0.95, 0.30, 0.30, 1.0],
        [0.95, 0.75, 0.20, 1.0],
        [0.30, 0.85, 0.40, 1.0],
    ];
    for (i, c) in dot_colors.iter().enumerate() {
        tree.add_child(
            panel,
            Node::rect()
                .pos(40.0 + i as f32 * 22.0, dot_y)
                .size(14.0, 14.0)
                .color(*c)
                .radius(7.0)
                .build(),
        );
    }

    // Swatch grid: 8 cols × 5 rows = 40 cards inside an inner row container.
    let grid_origin_x = 32.0;
    let grid_origin_y = 110.0;
    let cols = 8usize;
    let rows = 5usize;
    let cell_w = 110.0;
    let cell_h = 80.0;
    let gap = 12.0;
    let row = tree.add_child(
        panel,
        Node::rect()
            .pos(grid_origin_x, grid_origin_y)
            .size(
                cols as f32 * cell_w + (cols as f32 - 1.0) * gap,
                rows as f32 * cell_h + (rows as f32 - 1.0) * gap,
            )
            .rgba(0.0, 0.0, 0.0, 0.0) // invisible group container
            .build(),
    );

    for r in 0..rows {
        for c in 0..cols {
            let t = (r * cols + c) as f32 / ((rows * cols) as f32);
            let hue = t * 6.2831;
            let color = hsv_to_rgb(hue, 0.55, 0.95);
            tree.add_child(
                row,
                Node::rect()
                    .pos(c as f32 * (cell_w + gap), r as f32 * (cell_h + gap))
                    .size(cell_w, cell_h)
                    .rgba(color[0], color[1], color[2], 1.0)
                    .radius(14.0)
                    .border(1.0, [1.0, 1.0, 1.0, 0.20])
                    .shadow([0.0, 4.0], 8.0, [0.0, 0.0, 0.0, 1.0], 0.30)
                    .build(),
            );
        }
    }

    // Hero accent rect under the grid.
    let hero = tree.add_child(
        panel,
        Node::rect()
            .pos(32.0, grid_origin_y + rows as f32 * (cell_h + gap) + 18.0)
            .size(380.0, 70.0)
            .rgba(0.95, 0.25, 0.55, 1.0)
            .radius(20.0)
            .border(2.0, [1.0, 1.0, 1.0, 0.85])
            .shadow([0.0, 10.0], 22.0, [0.95, 0.25, 0.55, 1.0], 0.45)
            .build(),
    );

    (tree, DemoIds { hero })
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = v * s;
    let h6 = (h / 1.0471975) % 6.0;
    let x = c * (1.0 - (h6 % 2.0 - 1.0).abs());
    let (r, g, b) = match h6 as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r + m, g + m, b + m]
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        // Reactive: only redraw on actual events, idle = 0% CPU.
        event_loop.set_control_flow(ControlFlow::Wait);

        let attrs = Window::default_attributes()
            .with_title(self.config.title.clone())
            .with_transparent(true)
            .with_decorations(self.config.decorations)
            .with_resizable(true)
            .with_blur(true)
            .with_visible(false)
            .with_inner_size(winit::dpi::PhysicalSize::new(
                self.config.width,
                self.config.height,
            ));

        let window = event_loop
            .create_window(attrs)
            .expect("failed to create window");
        let window_arc: WindowArc = Arc::new(window);

        let gpu = GpuContext::new(Arc::clone(&window_arc));
        self.gpu = Some(gpu);
        // First flush walks the dirty tree built in App::new and uploads.
        self.flush_tree();
        log::info!(
            "first upload: {} instances ({} bytes)",
            self.instances.len(),
            self.instances.len() * std::mem::size_of::<ShapeInstance>()
        );
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.render_frame();
        }
        window_arc.set_visible(true);
        window_arc.request_redraw();

        if std::env::var_os("FROSTIFY_AUTOCAPTURE").is_some() {
            self.autocapture();
            if std::env::var_os("FROSTIFY_AUTOCAPTURE_TOGGLE").is_some() {
                // Drive the demo signal → dirty → flush → render → capture path.
                self.hero_lit.set(true);
                self.tree
                    .set_color(self.demo.hero, [0.20, 0.95, 0.55, 1.0]);
                let mask_before = self.tree.dirty();
                let flushed = self.flush_tree();
                log::info!(
                    "toggle: hero_lit.version={} dirty_mask=0x{:x} flushed={}",
                    self.hero_lit.version(),
                    mask_before,
                    flushed
                );
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.render_frame();
                }
                self.autocapture();
            }
            event_loop.exit();
        }

        self.window = Some(window_arc);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                log::info!("close requested");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                gpu.resize(size.width, size.height);
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                gpu.render_frame();
            }
            WindowEvent::Occluded(occluded) => {
                if !occluded {
                    if let Some(w) = &self.window {
                        w.request_redraw();
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                ..
            } => match code {
                KeyCode::F2 => {
                    let (rgba, w, h) = gpu.capture_rgba();
                    let path = debug::screenshot_path(&self.config.capture_dir);
                    match debug::save_png(&path, &rgba, w, h) {
                        Ok(()) => log::info!("screenshot saved: {}", path.display()),
                        Err(e) => log::error!("screenshot failed: {e}"),
                    }
                }
                KeyCode::F5 => {
                    self.tree.mark_all_dirty();
                    if self.flush_tree() {
                        if let Some(w) = &self.window {
                            w.request_redraw();
                        }
                    }
                }
                KeyCode::Space => {
                    let lit = !self.hero_lit.get();
                    self.hero_lit.set(lit);
                    let color = if lit {
                        [0.20, 0.95, 0.55, 1.0]
                    } else {
                        [0.95, 0.25, 0.55, 1.0]
                    };
                    self.tree.set_color(self.demo.hero, color);
                    if self.flush_tree() {
                        if let Some(w) = &self.window {
                            w.request_redraw();
                        }
                    }
                }
                KeyCode::Escape => event_loop.exit(),
                _ => {}
            },
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // M6 will set ControlFlow::WaitUntil here for animation deadlines.
    }
}
