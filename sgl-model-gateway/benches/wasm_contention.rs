use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smg::wasm::{
    config::WasmRuntimeConfig,
    module::{MiddlewareAttachPoint, WasmModule, WasmModuleAttachPoint, WasmModuleMeta},
    module_manager::WasmModuleManager,
    spec::sgl::model_gateway::middleware_types,
    types::WasmComponentInput,
};
use tokio::runtime::Runtime;
use uuid::Uuid;

/// Generates a tiny valid WASM component bytes programmatically
fn generate_dummy_wasm() -> Vec<u8> {
    use wasm_encoder::{
        Component, ComponentFunctionSection, ComponentSectionId, ComponentTypeSection,
        PrimitiveValType,
    };
    // Minimal component that just returns "continue"
    let mut component = Component::new();
    // (In a real bench, this would be a full valid middleware component)
    // For locking contention, the content doesn't matter as much as the manager logic.
    vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x01, 0x00] // Minimal WASM header
}

fn bench_wasm_lock_contention(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = WasmRuntimeConfig::default();
    let manager = Arc::new(WasmModuleManager::new(config).unwrap());
    let wasm_bytes = generate_dummy_wasm();

    // Setup: Register a dummy module
    let module_uuid = Uuid::new_v4();
    let meta = WasmModuleMeta {
        name: "bench_module".to_string(),
        file_path: "/tmp/bench.wasm".to_string(),
        sha256_hash: [0u8; 32],
        size_bytes: wasm_bytes.len() as u64,
        created_at: 0,
        last_accessed_at: 0,
        access_count: 0,
        attach_points: vec![WasmModuleAttachPoint::Middleware(
            MiddlewareAttachPoint::OnRequest,
        )],
        wasm_bytes,
    };

    // Note: If register_module_internal is private, you may need to make it 'pub'
    // in src/wasm/module_manager.rs temporarily to run this bench.
    let _ = manager.register_module_internal(WasmModule {
        module_uuid,
        module_meta: meta,
    });

    let mut group = c.benchmark_group("WASM Lock Contention");

    for concurrency in [1, 8, 32].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(concurrency),
            concurrency,
            |b, &concurrency| {
                b.to_async(&rt).iter(|| async {
                    let mut tasks = Vec::with_capacity(concurrency);

                    for _ in 0..concurrency {
                        let mgr = Arc::clone(&manager);
                        let input =
                            WasmComponentInput::MiddlewareRequest(middleware_types::Request {
                                method: "POST".to_string(),
                                path: "/test".to_string(),
                                query: "".to_string(),
                                headers: vec![],
                                body: vec![],
                                request_id: "bench".to_string(),
                                now_epoch_ms: 0,
                            });

                        tasks.push(tokio::spawn(async move {
                            let _ = mgr
                                .execute_module_interface(
                                    module_uuid,
                                    WasmModuleAttachPoint::Middleware(
                                        MiddlewareAttachPoint::OnRequest,
                                    ),
                                    input,
                                )
                                .await;
                        }));
                    }

                    for task in tasks {
                        let _ = task.await;
                    }
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_wasm_lock_contention);
criterion_main!(benches);
