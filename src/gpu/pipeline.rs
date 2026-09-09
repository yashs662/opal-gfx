//! Pipeline construction for the shape passes.
//!
//! Two render pipelines share one shader module:
//!   - `opaque_pipeline`: fs_opaque, targets `BACKDROP_FORMAT`. Only uses
//!     bind group 0 (frame + instances). Drawn into the backdrop texture.
//!   - `final_pipeline`: fs_main, targets the swapchain format. Uses
//!     bind group 0 + bind group 1 (backdrop pyramid + sampler) +
//!     glyph and image atlases. Drawn to the surface; glass instances
//!     sample the backdrop via dual-filter upsample at a per-instance LOD.

use wgpu::util::DeviceExt;

use super::blur::BACKDROP_FORMAT;
use super::instance::{FrameUniform, ShapeInstance};

pub struct ShapePipeline {
    pub opaque_pipeline: wgpu::RenderPipeline,
    pub final_pipeline: wgpu::RenderPipeline,
    /// Resets the bound scissor rect to transparent (damage rasters).
    pub clear_pipeline: wgpu::RenderPipeline,
    pub shape_bgl: wgpu::BindGroupLayout,
    pub glass_bgl: wgpu::BindGroupLayout,
    pub frame_buffer: wgpu::Buffer,
}

impl ShapePipeline {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        glyph_bgl: &wgpu::BindGroupLayout,
        image_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        // The shape shader is assembled from `shape.wgsl` plus every
        // registered effect (bodies + generated dispatch) — see
        // `gpu::effects`.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("opal.shape shader"),
            source: wgpu::ShaderSource::Wgsl(
                super::effects::compose_shader(include_str!("shaders/shape.wgsl")).into(),
            ),
        });

        let shape_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("opal.shape bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<FrameUniform>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<ShapeInstance>() as u64,
                        ),
                    },
                    count: None,
                },
                // Effect params (see `gpu::effects`): one flat arena
                // shared by every effect node in the frame. Visible to the
                // vertex stage too — a text effect places its glyph's quad
                // there from these same params.
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(4),
                    },
                    count: None,
                },
            ],
        });

        let glass_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("opal.shape.glass bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Opaque pipeline layout: groups 0 (shape) + 2 (glyph) + 3
        // (image). Group 1 stays unbound — `fs_opaque` doesn't sample
        // the blurred backdrop (it writes its source). Glyph + image
        // atlases are independent textures so they're safe to bind in
        // both passes; this lets text and images participate in the
        // backdrop and appear blurred behind glass panels.
        let opaque_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("opal.shape.opaque pl"),
            bind_group_layouts: &[Some(&shape_bgl), None, Some(glyph_bgl), Some(image_bgl)],
            immediate_size: 0,
        });

        // Final pipeline layout: bind groups 0..=3.
        let final_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("opal.shape.final pl"),
            bind_group_layouts: &[
                Some(&shape_bgl),
                Some(&glass_bgl),
                Some(glyph_bgl),
                Some(image_bgl),
            ],
            immediate_size: 0,
        });

        let blend = Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        });

        let make_pipeline = |label: &str,
                             layout: &wgpu::PipelineLayout,
                             fs_entry: &'static str,
                             format: wgpu::TextureFormat|
         -> wgpu::RenderPipeline {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs_entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        let opaque_pipeline = make_pipeline(
            "opal.shape.opaque",
            &opaque_layout,
            "fs_opaque",
            BACKDROP_FORMAT,
        );
        let final_pipeline =
            make_pipeline("opal.shape.final", &final_layout, "fs_main", surface_format);

        // Scissored clear (see `scissor_clear.wgsl`) — used by damage-rect
        // layer rasters, which keep the rest of the cached texture.
        let clear_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("opal.scissor-clear shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/scissor_clear.wgsl").into()),
        });
        let clear_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("opal.scissor-clear layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let clear_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("opal.scissor-clear pipeline"),
            layout: Some(&clear_layout),
            vertex: wgpu::VertexState {
                module: &clear_shader,
                entry_point: Some("vs_clear"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &clear_shader,
                entry_point: Some("fs_clear"),
                targets: &[Some(wgpu::ColorTargetState {
                    // Layer raster textures carry the surface format.
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let frame_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("opal.frame ubo"),
            contents: bytemuck::bytes_of(&FrameUniform::new(
                [1.0, 1.0],
                0.0,
                0.0,
                Default::default(),
            )),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            clear_pipeline,
            opaque_pipeline,
            final_pipeline,
            shape_bgl,
            glass_bgl,
            frame_buffer,
        }
    }
}
