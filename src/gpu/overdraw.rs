//! Overdraw heatmap resources.
//!
//! Two pipelines:
//!   - `count_pipeline`: same vertex shader as the shape pipeline, but a
//!     fragment shader that emits the per-fragment SDF coverage into the
//!     R channel of an Rgba16Float accumulator with additive blending.
//!     Driven from the same instance buffer, so what you see in the
//!     scene is what you measure here.
//!   - `compose_pipeline`: fullscreen quad that samples the accumulator
//!     and remaps the per-pixel count through a 5-stop heatmap.
//!
//! Toggled by `GpuContext::set_overdraw`. When off, none of these
//! resources cost anything per frame — the whole pipeline branch is
//! gated in `encode_frame`.

pub const OVERDRAW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub struct OverdrawResources {
    pub count_view: wgpu::TextureView,
    count_tex: wgpu::Texture,
    pub sampler: wgpu::Sampler,

    pub count_pipeline: wgpu::RenderPipeline,
    pub compose_pipeline: wgpu::RenderPipeline,

    pub compose_bgl: wgpu::BindGroupLayout,
    pub compose_bg: wgpu::BindGroup,

    width: u32,
    height: u32,
}

impl OverdrawResources {
    /// Current resolution of the count texture.
    pub fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

impl OverdrawResources {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
        shape_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let (count_tex, count_view) = make_count_tex(device, width, height);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("opal.overdraw sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // Count pipeline shares the shape bind group layout (frame +
        // instance buffer). It samples nothing, so no group 1.
        let count_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("opal.overdraw shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/overdraw.wgsl").into()),
        });

        let count_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("opal.overdraw.count pl"),
            bind_group_layouts: &[Some(shape_bgl)],
            immediate_size: 0,
        });

        let additive = Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        });

        let count_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("opal.overdraw.count"),
            layout: Some(&count_layout),
            vertex: wgpu::VertexState {
                module: &count_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &count_shader,
                entry_point: Some("fs_count"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: OVERDRAW_FORMAT,
                    blend: additive,
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
        });

        // Compose pipeline: fullscreen quad sampling the count texture.
        let compose_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("opal.overdraw.compose shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/overdraw_compose.wgsl").into()),
        });

        let compose_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("opal.overdraw.compose bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let compose_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("opal.overdraw.compose pl"),
            bind_group_layouts: &[Some(&compose_bgl)],
            immediate_size: 0,
        });

        let compose_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("opal.overdraw.compose"),
            layout: Some(&compose_layout),
            vertex: wgpu::VertexState {
                module: &compose_shader,
                entry_point: Some("vs_compose"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &compose_shader,
                entry_point: Some("fs_compose"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
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
        });

        let compose_bg = make_compose_bg(device, &compose_bgl, &count_view, &sampler);

        Self {
            count_view,
            count_tex,
            sampler,
            count_pipeline,
            compose_pipeline,
            compose_bgl,
            compose_bg,
            width,
            height,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }
        let (tex, view) = make_count_tex(device, width, height);
        self.count_tex = tex;
        self.count_view = view;
        self.compose_bg =
            make_compose_bg(device, &self.compose_bgl, &self.count_view, &self.sampler);
        self.width = width;
        self.height = height;
    }
}

fn make_count_tex(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("opal.overdraw.count"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: OVERDRAW_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn make_compose_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("opal.overdraw.compose bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}
