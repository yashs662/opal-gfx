//! CPU-rendered startup splash — fills the cold-start gap.
//!
//! On a cold launch the GPU back-end (d3d12 / dxgi + the driver UMD on
//! Windows) can take ~2 s to load before [`crate::gpu::GpuContext::new`]
//! returns and the real window can be shown. That whole time the app has
//! nothing on screen. This module paints a tiny logo + wordmark bitmap on
//! the CPU (no GPU at all — resvg for the mark, cosmic-text for the
//! wordmark, both already in the engine) and shows it in a borderless,
//! always-on-top window **before** the blocking GPU init. The splash sits
//! there untouched (no event pump needed — the OS compositor holds its
//! pixels) for the duration of the init, then [`Splash::close`] tears it
//! down right after the real window becomes visible.
//!
//! Platform notes:
//! - **Windows**: a layered window (`UpdateLayeredWindow` + premultiplied
//!   BGRA) gives true per-pixel alpha, so only the artwork is visible —
//!   it floats transparently over the desktop with no backing rectangle.
//! - **Other**: a borderless winit window presented via `softbuffer`.
//!   softbuffer has no per-pixel-alpha path, so the bitmap is composited
//!   over an opaque brand-background panel. These platforms don't exhibit
//!   the multi-second GPU-DLL cold load that motivates the splash, so the
//!   panel is a best-effort nicety rather than a gap-filler.
//!
//! The bitmap build is a few milliseconds and runs *before* the ~2 s GPU
//! init, so the window appears **sooner**, never later — startup is not
//! slowed. Sizes are logical px scaled by the target monitor's DPI, so the
//! splash is crisp and consistently sized across 1× / 1.5× / 2× displays.

use crate::text::TextResources;

/// Declarative splash content. All sizes are logical px (DPI-scaled at
/// build time). Built by the host (it owns the brand assets) and handed to
/// the engine via [`crate::App::splash`].
#[derive(Clone, Debug)]
pub struct SplashConfig {
    /// Brand mark, as raw SVG bytes (rendered with its own gradient).
    pub logo_svg: Vec<u8>,
    /// Wordmark drawn beside the mark (e.g. `"Opal"`).
    pub wordmark: String,
    /// Mark edge length, logical px.
    pub logo_px: f32,
    /// Wordmark font size, logical px.
    pub wordmark_px: f32,
    /// Gap between mark and wordmark, logical px.
    pub gap_px: f32,
    /// Wordmark color, linear-ish `[r, g, b, a]` in 0..1 (matches the
    /// engine's color convention).
    pub wordmark_color: [f32; 4],
    /// Opaque panel background used only on the non-Windows softbuffer
    /// path (which can't do per-pixel alpha). Ignored on Windows.
    pub bg_color: [f32; 4],
}

/// A built RGBA splash bitmap — straight (non-premultiplied) alpha,
/// row-major, top-down, `w * h * 4` bytes. Physical px.
pub struct SplashBitmap {
    pub w: u32,
    pub h: u32,
    pub rgba: Vec<u8>,
}

impl SplashBitmap {
    /// Rasterize the logo mark + wordmark into a single tight RGBA bitmap
    /// at `scale` (the target monitor's DPI factor). Returns `None` if the
    /// mark fails to rasterize to a usable size.
    pub fn build(cfg: &SplashConfig, text: &mut TextResources, scale: f32) -> Option<Self> {
        let scale = scale.max(0.1);
        let logo_phys = (cfg.logo_px * scale).round().max(1.0) as u32;
        let gap_phys = (cfg.gap_px * scale).round().max(0.0) as u32;

        // Mark: resvg → straight RGBA, square.
        let logo_rgba = crate::svg::rasterize_svg(&cfg.logo_svg, logo_phys);
        if logo_rgba.len() != (logo_phys * logo_phys * 4) as usize {
            return None;
        }

        // Wordmark: shape + rasterize each glyph (R8 coverage), tinted.
        let (word_w, word_h, word_rgba) = build_wordmark(cfg, text, scale);

        let total_w = logo_phys + if word_w > 0 { gap_phys + word_w } else { 0 };
        let total_h = logo_phys.max(word_h).max(1);
        let mut canvas = vec![0u8; (total_w * total_h * 4) as usize];

        // Mark, vertically centered at x=0.
        let logo_y = (total_h - logo_phys) / 2;
        blit_over(
            &mut canvas,
            total_w,
            total_h,
            &logo_rgba,
            logo_phys,
            logo_phys,
            0,
            logo_y,
        );
        // Wordmark, vertically centered after the gap.
        if word_w > 0 {
            let word_x = logo_phys + gap_phys;
            let word_y = (total_h.saturating_sub(word_h)) / 2;
            blit_over(
                &mut canvas,
                total_w,
                total_h,
                &word_rgba,
                word_w,
                word_h,
                word_x,
                word_y,
            );
        }

        Some(Self {
            w: total_w,
            h: total_h,
            rgba: canvas,
        })
    }

    /// Premultiplied BGRA, the byte order `UpdateLayeredWindow` expects
    /// for an `AC_SRC_ALPHA` blend.
    #[cfg(windows)]
    fn bgra_premultiplied(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.rgba.len()];
        for (s, d) in self.rgba.chunks_exact(4).zip(out.chunks_exact_mut(4)) {
            let (r, g, b, a) = (s[0] as u32, s[1] as u32, s[2] as u32, s[3] as u32);
            d[0] = (b * a / 255) as u8;
            d[1] = (g * a / 255) as u8;
            d[2] = (r * a / 255) as u8;
            d[3] = a as u8;
        }
        out
    }

    /// Composite over an opaque background into softbuffer's `0RGB` u32
    /// format (alpha byte ignored by softbuffer).
    #[cfg(not(windows))]
    fn to_u32_over(&self, bg: [f32; 4]) -> Vec<u32> {
        let br = (bg[0] * 255.0) as u32;
        let bgc = (bg[1] * 255.0) as u32;
        let bb = (bg[2] * 255.0) as u32;
        self.rgba
            .chunks_exact(4)
            .map(|p| {
                let a = p[3] as u32;
                let r = (p[0] as u32 * a + br * (255 - a)) / 255;
                let g = (p[1] as u32 * a + bgc * (255 - a)) / 255;
                let b = (p[2] as u32 * a + bb * (255 - a)) / 255;
                (r << 16) | (g << 8) | b
            })
            .collect()
    }
}

/// Rasterize the wordmark to a tight straight-alpha RGBA buffer. Returns
/// `(0, 0, vec![])` when the string shapes to nothing.
fn build_wordmark(cfg: &SplashConfig, text: &mut TextResources, scale: f32) -> (u32, u32, Vec<u8>) {
    if cfg.wordmark.is_empty() {
        return (0, 0, Vec::new());
    }
    let size = cfg.wordmark_px * scale;
    let line_h = size * 1.25;
    let glyphs = text.shape(&cfg.wordmark, size, line_h);

    // Collect rasters + the union bounding box (pen-space, per the
    // text-module convention: glyph top-left = pen + (left, -top)).
    let mut placed = Vec::new();
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for g in &glyphs {
        let Some(r) = text.rasterize(g.cache_key) else {
            continue;
        };
        if r.width == 0 || r.height == 0 {
            continue;
        }
        let gx = g.x + r.left;
        let gy = g.y - r.top;
        min_x = min_x.min(gx);
        min_y = min_y.min(gy);
        max_x = max_x.max(gx + r.width as i32);
        max_y = max_y.max(gy + r.height as i32);
        placed.push((gx, gy, r));
    }
    if placed.is_empty() {
        return (0, 0, Vec::new());
    }
    let w = (max_x - min_x) as u32;
    let h = (max_y - min_y) as u32;
    let mut canvas = vec![0u8; (w * h * 4) as usize];
    let (cr, cg, cb, ca) = (
        (cfg.wordmark_color[0] * 255.0) as u8,
        (cfg.wordmark_color[1] * 255.0) as u8,
        (cfg.wordmark_color[2] * 255.0) as u8,
        cfg.wordmark_color[3],
    );
    for (gx, gy, r) in &placed {
        let ox = (gx - min_x) as u32;
        let oy = (gy - min_y) as u32;
        for py in 0..r.height {
            for px in 0..r.width {
                let cov = r.data[(py * r.width + px) as usize];
                if cov == 0 {
                    continue;
                }
                let dx = ox + px;
                let dy = oy + py;
                let di = ((dy * w + dx) * 4) as usize;
                canvas[di] = cr;
                canvas[di + 1] = cg;
                canvas[di + 2] = cb;
                canvas[di + 3] = (cov as f32 * ca) as u8;
            }
        }
    }
    (w, h, canvas)
}

/// Alpha-over blit of a straight-RGBA `src` onto a straight-RGBA `dst`.
/// Splash regions don't overlap, but a proper over keeps soft edges clean
/// where they meet.
fn blit_over(
    dst: &mut [u8],
    dst_w: u32,
    dst_h: u32,
    src: &[u8],
    src_w: u32,
    src_h: u32,
    at_x: u32,
    at_y: u32,
) {
    for sy in 0..src_h {
        let dy = at_y + sy;
        if dy >= dst_h {
            break;
        }
        for sx in 0..src_w {
            let dx = at_x + sx;
            if dx >= dst_w {
                break;
            }
            let si = ((sy * src_w + sx) * 4) as usize;
            let sa = src[si + 3] as u32;
            if sa == 0 {
                continue;
            }
            let di = ((dy * dst_w + dx) * 4) as usize;
            if sa == 255 || dst[di + 3] == 0 {
                dst[di..di + 4].copy_from_slice(&src[si..si + 4]);
                continue;
            }
            let ia = 255 - sa;
            for c in 0..3 {
                dst[di + c] = ((src[si + c] as u32 * sa + dst[di + c] as u32 * ia) / 255) as u8;
            }
            dst[di + 3] = (sa + dst[di + 3] as u32 * ia / 255) as u8;
        }
    }
}

/// A live splash window. Dropping or [`Self::close`]-ing it removes the
/// splash. Held by the shell across the blocking GPU init.
pub struct Splash {
    // Held only for its `Drop` (which tears the OS window down). Never
    // read by name — the resource lifetime *is* the use.
    #[allow(dead_code)]
    #[cfg(windows)]
    inner: windows_impl::WinSplash,
    #[allow(dead_code)]
    #[cfg(not(windows))]
    inner: other_impl::SbSplash,
}

impl Splash {
    /// Build + show the splash centered at the given **physical** screen
    /// position (top-left of the bitmap). `event_loop` is only used on
    /// non-Windows platforms (to create the winit presenter window);
    /// Windows creates a raw layered window directly. Returns `None` on
    /// any platform failure — the splash is purely cosmetic, so callers
    /// ignore the error and continue startup.
    #[cfg(windows)]
    pub fn show(bmp: &SplashBitmap, screen_x: i32, screen_y: i32) -> Option<Self> {
        windows_impl::WinSplash::show(bmp, screen_x, screen_y).map(|inner| Self { inner })
    }

    #[cfg(not(windows))]
    pub fn show(
        event_loop: &winit::event_loop::ActiveEventLoop,
        cfg: &SplashConfig,
        bmp: &SplashBitmap,
        screen_x: i32,
        screen_y: i32,
        scale: f64,
    ) -> Option<Self> {
        other_impl::SbSplash::show(event_loop, cfg, bmp, screen_x, screen_y, scale)
            .map(|inner| Self { inner })
    }

    /// Tear the splash down. Called right after the real window is shown.
    pub fn close(self) {
        // Drop runs the platform teardown.
    }
}

// ---------------------------------------------------------------------------
// Windows: layered window with true per-pixel alpha.
// ---------------------------------------------------------------------------
#[cfg(windows)]
mod windows_impl {
    use super::SplashBitmap;
    use std::sync::Once;
    use windows::Win32::Foundation::{COLORREF, HWND, LPARAM, LRESULT, POINT, SIZE, WPARAM};
    use windows::Win32::Graphics::Gdi::{
        AC_SRC_ALPHA, AC_SRC_OVER, BITMAPINFO, BITMAPINFOHEADER, BLENDFUNCTION, CreateCompatibleDC,
        CreateDIBSection, DIB_RGB_COLORS, DeleteDC, DeleteObject, GetDC, HBITMAP, HDC, ReleaseDC,
        SelectObject,
    };
    use windows::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows::Win32::UI::WindowsAndMessaging::{
        CreateWindowExW, DefWindowProcW, DestroyWindow, HMENU, RegisterClassW, SW_SHOWNOACTIVATE,
        ShowWindow, ULW_ALPHA, UpdateLayeredWindow, WNDCLASSW, WS_EX_LAYERED, WS_EX_NOACTIVATE,
        WS_EX_TOOLWINDOW, WS_EX_TOPMOST, WS_POPUP,
    };
    use windows::core::{PCWSTR, w};

    const CLASS: PCWSTR = w!("OpalSplashWindow");
    static REGISTER: Once = Once::new();

    unsafe extern "system" fn wndproc(h: HWND, m: u32, w: WPARAM, l: LPARAM) -> LRESULT {
        unsafe { DefWindowProcW(h, m, w, l) }
    }

    fn ensure_class() {
        REGISTER.call_once(|| unsafe {
            let hinst = GetModuleHandleW(None).unwrap_or_default();
            let wc = WNDCLASSW {
                lpfnWndProc: Some(wndproc),
                hInstance: hinst.into(),
                lpszClassName: CLASS,
                ..Default::default()
            };
            RegisterClassW(&wc);
        });
    }

    pub struct WinSplash {
        hwnd: HWND,
        hdc_mem: HDC,
        hbitmap: HBITMAP,
    }

    impl WinSplash {
        pub fn show(bmp: &SplashBitmap, x: i32, y: i32) -> Option<Self> {
            ensure_class();
            let (w, h) = (bmp.w as i32, bmp.h as i32);
            let premul = bmp.bgra_premultiplied();
            unsafe {
                let hinst = GetModuleHandleW(None).ok();
                let hwnd = CreateWindowExW(
                    WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_TOPMOST | WS_EX_NOACTIVATE,
                    CLASS,
                    w!("Opal"),
                    WS_POPUP,
                    x,
                    y,
                    w,
                    h,
                    None,
                    None::<HMENU>,
                    hinst.map(|m| m.into()),
                    None,
                )
                .ok()?;

                let screen_dc = GetDC(None);
                let hdc_mem = CreateCompatibleDC(Some(screen_dc));

                let bmi = BITMAPINFO {
                    bmiHeader: BITMAPINFOHEADER {
                        biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                        biWidth: w,
                        biHeight: -h, // top-down
                        biPlanes: 1,
                        biBitCount: 32,
                        biCompression: 0, // BI_RGB
                        ..Default::default()
                    },
                    ..Default::default()
                };
                let mut bits: *mut core::ffi::c_void = std::ptr::null_mut();
                let hbitmap = match CreateDIBSection(
                    Some(screen_dc),
                    &bmi,
                    DIB_RGB_COLORS,
                    &mut bits,
                    None,
                    0,
                ) {
                    Ok(b) if !bits.is_null() => b,
                    _ => {
                        ReleaseDC(None, screen_dc);
                        let _ = DeleteDC(hdc_mem);
                        let _ = DestroyWindow(hwnd);
                        return None;
                    }
                };
                std::ptr::copy_nonoverlapping(premul.as_ptr(), bits as *mut u8, premul.len());
                let old = SelectObject(hdc_mem, hbitmap.into());

                let blend = BLENDFUNCTION {
                    BlendOp: AC_SRC_OVER as u8,
                    BlendFlags: 0,
                    SourceConstantAlpha: 255,
                    AlphaFormat: AC_SRC_ALPHA as u8,
                };
                let pos = POINT { x, y };
                let size = SIZE { cx: w, cy: h };
                let src = POINT { x: 0, y: 0 };
                let ok = UpdateLayeredWindow(
                    hwnd,
                    Some(screen_dc),
                    Some(&pos),
                    Some(&size),
                    Some(hdc_mem),
                    Some(&src),
                    COLORREF(0),
                    Some(&blend),
                    ULW_ALPHA,
                )
                .is_ok();
                ReleaseDC(None, screen_dc);

                if !ok {
                    SelectObject(hdc_mem, old);
                    let _ = DeleteObject(hbitmap.into());
                    let _ = DeleteDC(hdc_mem);
                    let _ = DestroyWindow(hwnd);
                    return None;
                }

                let _ = ShowWindow(hwnd, SW_SHOWNOACTIVATE);
                Some(Self {
                    hwnd,
                    hdc_mem,
                    hbitmap,
                })
            }
        }
    }

    impl Drop for WinSplash {
        fn drop(&mut self) {
            unsafe {
                let _ = DestroyWindow(self.hwnd);
                let _ = DeleteObject(self.hbitmap.into());
                let _ = DeleteDC(self.hdc_mem);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Non-Windows: borderless winit window presented via softbuffer (opaque).
// ---------------------------------------------------------------------------
#[cfg(not(windows))]
mod other_impl {
    use super::{SplashBitmap, SplashConfig};
    use std::num::NonZeroU32;
    use std::sync::Arc;
    use winit::dpi::{LogicalSize, PhysicalPosition};
    use winit::event_loop::ActiveEventLoop;
    use winit::window::{Window, WindowLevel};

    pub struct SbSplash {
        // Drop order (declaration order): surface → context → window. All
        // three hold an Arc<Window>; the OS window closes only when the
        // last clone drops, so every holder must live in this struct —
        // leaking any of them keeps the always-on-top splash up forever.
        _surface: softbuffer::Surface<Arc<Window>, Arc<Window>>,
        _context: softbuffer::Context<Arc<Window>>,
        _window: Arc<Window>,
    }

    impl SbSplash {
        pub fn show(
            event_loop: &ActiveEventLoop,
            cfg: &SplashConfig,
            bmp: &SplashBitmap,
            x: i32,
            y: i32,
            scale: f64,
        ) -> Option<Self> {
            let logical = LogicalSize::new(bmp.w as f64 / scale, bmp.h as f64 / scale);
            let attrs = Window::default_attributes()
                .with_title("Opal")
                .with_decorations(false)
                .with_resizable(false)
                .with_active(false)
                .with_window_level(WindowLevel::AlwaysOnTop)
                .with_position(PhysicalPosition::new(x, y))
                .with_inner_size(logical);
            let window = Arc::new(event_loop.create_window(attrs).ok()?);

            let context = softbuffer::Context::new(window.clone()).ok()?;
            let mut surface = softbuffer::Surface::new(&context, window.clone()).ok()?;
            surface
                .resize(NonZeroU32::new(bmp.w)?, NonZeroU32::new(bmp.h)?)
                .ok()?;
            let pixels = bmp.to_u32_over(cfg.bg_color);
            let mut buf = surface.buffer_mut().ok()?;
            buf.copy_from_slice(&pixels);
            buf.present().ok()?;

            Some(Self {
                _surface: surface,
                _context: context,
                _window: window,
            })
        }
    }
}
