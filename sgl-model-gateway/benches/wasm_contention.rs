use std::sync::{atomic::AtomicU64, Arc};

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

// Minimal valid WASM component header
const DUMMY_WASM: &[u8] = &[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x01, 0x00];

fn bench_wasm_lock_contention(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = WasmRuntimeConfig::default();
    let manager = Arc::new(WasmModuleManager::new(config).unwrap());

    let module_uuid = Uuid::new_v4();
    let meta = WasmModuleMeta {
        name: "bench_module".to_string(),
        file_path: "/tmp/bench.wasm".to_string(),
        sha256_hash: [0u8; 32],
        size_bytes: DUMMY_WASM.len() as u64,
        created_at: 0,
        // These fields will be Atomics after the fix
        last_accessed_at: 0.into(),
        access_count: 0.into(),
        attach_points: vec![WasmModuleAttachPoint::Middleware(
            MiddlewareAttachPoint::OnRequest,
        )],
        wasm_bytes: DUMMY_WASM.to_vec().into(),
    };

    // Register the module once
    manager
        .register_module_internal(WasmModule {
            module_uuid,
            module_meta: meta,
        })
        .unwrap();

    let mut group = c.benchmark_group("WASM Lock Contention");

    // Benchmark varying levels of concurrent pressure on the global manager
    for concurrency in [1, 8, 32].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(concurrency),
            concurrency,
            |b, &concurrency| {
                b.to_async(&rt).iter(|| async {
                    let mut tasks = Vec::with_capacity(concurrency);

                    for i in 0..concurrency {
                        let mgr = Arc::clone(&manager);
                        let input =
                            WasmComponentInput::MiddlewareRequest(middleware_types::Request {
                                method: "POST".to_string(),
                                path: "/test".to_string(),
                                query: "".to_string(),
                                headers: vec![],
                                body: vec![],
                                request_id: format!("req-{}", i),
                                now_epoch_ms: 123456789,
                            });

                        tasks.push(tokio::spawn(async move {
                            // This is the hot path that triggers the lock contention
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
