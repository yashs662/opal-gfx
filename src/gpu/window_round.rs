//! Final window-corner round pass resources.
//!
//! The frame is composited into [`frame_view`](Self::frame_view) (an
//! offscreen, surface-format, surface-sized texture), then [`pipeline`] blits
//! it to the swapchain masking the four corners to a rounded rect — see
//! `shaders/window_round.wgsl`. Rounding the composited result *once* is what
//! lets stacked translucent content (glass over backdrop art) round cleanly
//! at the window corners without the back layer leaking through.

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RoundU {
    screen: [f32; 2],
    radius: f32,
    _pad: f32,
}

pub struct WindowRound {
    /// Render target the composite pass draws the whole frame into.
    frame_view: wgpu::TextureView,
    frame_tex: wgpu::Texture,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    bg: wgpu::BindGroup,
    width: u32,
    height: u32,
}

impl WindowRound {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let (frame_tex, frame_view) = make_frame_tex(device, width, height, surface_format);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("opal.window_round sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("opal.window_round uniform"),
            size: std::mem::size_of::<RoundU>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("opal.window_round shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/window_round.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("opal.window_round bgl"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("opal.window_round pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("opal.window_round"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
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
        let bg = make_bg(device, &bgl, &frame_view, &sampler, &uniform);
        Self {
            frame_view,
            frame_tex,
            sampler,
            uniform,
            pipeline,
            bgl,
            bg,
            width,
            height,
        }
    }

    pub fn frame_view(&self) -> &wgpu::TextureView {
        &self.frame_view
    }

    pub fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bg
    }

    /// Update the corner radius + surface size (physical px) for this frame.
    pub fn set_uniform(&self, queue: &wgpu::Queue, radius: f32) {
        let u = RoundU {
            screen: [self.width as f32, self.height as f32],
            radius: radius.max(0.0),
            _pad: 0.0,
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
    }

    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
    ) {
        if self.width == width && self.height == height {
            return;
        }
        let (tex, view) = make_frame_tex(device, width, height, surface_format);
        self.frame_tex = tex;
        self.frame_view = view;
        self.bg = make_bg(
            device,
            &self.bgl,
            &self.frame_view,
            &self.sampler,
            &self.uniform,
        );
        self.width = width;
        self.height = height;
    }

    pub fn memory_bytes(&self) -> u64 {
        (self.width as u64) * (self.height as u64) * 4
    }
}

fn make_frame_tex(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("opal.window_round.frame"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn make_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    uniform: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("opal.window_round bg"),
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
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform.as_entire_binding(),
            },
        ],
    })
}
