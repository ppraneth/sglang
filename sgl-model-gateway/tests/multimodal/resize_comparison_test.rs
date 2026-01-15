use std::time::Instant;

use ndarray::Array3;
// Use 'smg' as the crate name
use smg::multimodal::vision::transforms::{bicubic_resize as current_resize, to_tensor};

/// The "Legacy" implementation for comparison
fn bicubic_resize_legacy(tensor: &Array3<f32>, target_h: usize, target_w: usize) -> Array3<f32> {
    let (c, h, w) = (tensor.shape()[0], tensor.shape()[1], tensor.shape()[2]);
    let mut result = Array3::<f32>::zeros((c, target_h, target_w));
    let scale_h = h as f32 / target_h as f32;
    let scale_w = w as f32 / target_w as f32;

    for ch in 0..c {
        for y in 0..target_h {
            for x in 0..target_w {
                let src_y = (y as f32 + 0.5) * scale_h - 0.5;
                let src_x = (x as f32 + 0.5) * scale_w - 0.5;
                // Corrected: changed sgl_model_gateway to smg
                result[[ch, y, x]] = smg::multimodal::vision::transforms::bicubic_interpolate(
                    tensor, ch, src_y, src_x, h, w,
                );
            }
        }
    }
    result
}

#[test]
fn test_compare_resize_implementations() {
    let img_path = "tests/fixtures/images/large.jpg";
    let img = image::open(img_path).expect("Failed to load test image");
    let tensor = to_tensor(&img);

    let (t_h, t_w) = (336, 336);

    println!("\n--- Bicubic Resize Benchmark ---");
    println!(
        "Source: {}x{}, Target: {}x{}",
        img.width(),
        img.height(),
        t_w,
        t_h
    );

    // 1. Correctness
    let output_legacy = bicubic_resize_legacy(&tensor, t_h, t_w);
    let output_optimized = current_resize(&tensor, t_h, t_w);

    let diff = (&output_legacy - &output_optimized).mapv(|a| a.abs()).sum();
    assert!(diff < 1e-4, "Resize values differ! Error: {}", diff);
    println!("✅ Correctness: Outputs are identical.");

    // 2. Performance
    let iterations = 20;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = bicubic_resize_legacy(&tensor, t_h, t_w);
    }
    let avg_legacy = start.elapsed() / iterations;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = current_resize(&tensor, t_h, t_w);
    }
    let avg_optimized = start.elapsed() / iterations;

    println!("\nPerformance Metrics (Average over {} runs):", iterations);
    println!("- Legacy (Redundant): {:?}/image", avg_legacy);
    println!("- Optimized (LUT):    {:?}/image", avg_optimized);

    let speedup = avg_legacy.as_secs_f64() / avg_optimized.as_secs_f64();
    println!("\n🚀 Improvement: {:.2}x faster", speedup);
}
