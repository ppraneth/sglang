use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use smg::tokenizer::{
    mock::MockTokenizer,
    stop::{StopSequenceConfig, StopSequenceDecoder},
};

fn bench_stop_sequence_scaling(c: &mut Criterion) {
    let tokenizer = Arc::new(MockTokenizer::new());

    // We'll test with 1, 10, 50, and 100 stop sequences to see the O(M) impact
    let sequence_counts = vec![1, 10, 50, 100];

    let mut group = c.benchmark_group("StopSequence_Scaling");

    for count in sequence_counts {
        let mut config = StopSequenceConfig::default();
        for i in 0..count {
            config = config.with_stop_sequence(format!("STOP_PHRASE_{:03}", i));
        }

        // We process a sequence of tokens that don't match,
        // forcing the decoder to do full scans and partial match checks every time.
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, _| {
            b.iter(|| {
                let mut decoder =
                    StopSequenceDecoder::new(tokenizer.clone(), config.clone(), false);
                for &token in &tokens {
                    let _ = black_box(decoder.process_token(token).unwrap());
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_stop_sequence_scaling);
criterion_main!(benches);
