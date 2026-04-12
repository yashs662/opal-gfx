use bytemuck::{Pod, Zeroable};

/// GPU shape instance. Layout must match `ShapeInstance` in `shape.wgsl`.
/// std430-compatible. 112 bytes. WGSL `array<ShapeInstance>` stride is also
/// 112 (already 16-aligned), so the Rust struct must match exactly — do
/// **not** add trailing pad fields without also adding them on the WGSL
/// side, or every instance after the first will read garbage.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ShapeInstance {
    pub color: [f32; 4],
    pub border_color: [f32; 4],
    pub shadow_color: [f32; 4],
    pub border_radius: [f32; 4], // tl, tr, bl, br
    pub position: [f32; 2],      // top-left, pixels
    pub size: [f32; 2],          // pixels
    pub shadow_offset: [f32; 2],
    pub _pad0: [f32; 2],
    pub border_width: f32,
    pub shadow_blur: f32,
    pub shadow_opacity: f32,
    pub opacity: f32,
}

const _: () = assert!(std::mem::size_of::<ShapeInstance>() == 112);

impl Default for ShapeInstance {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0, 1.0],
            border_color: [0.0, 0.0, 0.0, 1.0],
            shadow_color: [0.0, 0.0, 0.0, 1.0],
            border_radius: [0.0; 4],
            position: [0.0; 2],
            size: [0.0; 2],
            shadow_offset: [0.0; 2],
            _pad0: [0.0; 2],
            border_width: 0.0,
            shadow_blur: 0.0,
            shadow_opacity: 0.0,
            opacity: 1.0,
        }
    }
}

impl ShapeInstance {
    pub fn rounded_rect(
        position: [f32; 2],
        size: [f32; 2],
        radius: f32,
        color: [f32; 4],
    ) -> Self {
        Self {
            color,
            position,
            size,
            border_radius: [radius; 4],
            ..Default::default()
        }
    }
}

/// Per-frame uniform: viewport size in physical pixels.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FrameUniform {
    pub screen_size: [f32; 2],
    pub _pad: [f32; 2],
}
