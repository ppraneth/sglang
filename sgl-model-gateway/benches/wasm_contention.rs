use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smg::wasm::{
    config::WasmRuntimeConfig,
    module::{WasmModule, WasmModuleAttachPoint, WasmModuleMeta, WasmModuleType},
    module_manager::WasmModuleManager,
    types::WasmComponentInput,
};
use tokio::runtime::Runtime;
use uuid::Uuid;

/// A minimal valid WASM component (empty) to avoid execution overhead
/// and focus purely on the manager's lock contention.
const DUMMY_WASM: &[u8] = include_bytes!("dummy.wasm"); // Ensure a small .wasm file exists or use a mock

fn bench_wasm_lock_contention(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = WasmRuntimeConfig::default();
    let manager = Arc::new(WasmModuleManager::new(config).unwrap());

    // Setup: Register a dummy module
    let module_uuid = Uuid::new_v4();
    let meta = WasmModuleMeta {
        name: "bench_module".to_string(),
        file_path: "/tmp/bench.wasm".to_string(),
        sha256_hash: [0u8; 32],
        size_bytes: DUMMY_WASM.len() as u64,
        created_at: 0,
        last_accessed_at: 0,
        access_count: 0,
        attach_points: vec![],
        wasm_bytes: DUMMY_WASM.to_vec(),
    };
    let module = WasmModule {
        module_uuid,
        module_meta: meta,
    };

    // Note: register_module_internal is pub(crate), so for this bench to work
    // it must be in the benches folder of the smg crate or the method must be public.
    manager.register_module_internal(module).unwrap();

    let mut group = c.benchmark_group("WASM Lock Contention");

    // Test with increasing levels of concurrency
    for concurrency in [1, 4, 8, 16, 32, 64].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(concurrency),
            concurrency,
            |b, &concurrency| {
                b.to_async(&rt).iter(|| async {
                    let mut tasks = Vec::with_capacity(concurrency);

                    for _ in 0..concurrency {
                        let mgr = Arc::clone(&manager);
                        let input = WasmComponentInput::MiddlewareRequest(Default::default());

                        tasks.push(tokio::spawn(async move {
                            // This call triggers the .write() lock in the current implementation
                            let _ = mgr
                                .execute_module_interface(
                                    module_uuid,
                                    WasmModuleAttachPoint::Middleware(
                                        smg::wasm::module::MiddlewareAttachPoint::OnRequest,
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
