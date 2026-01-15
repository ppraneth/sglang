use std::time::Instant;

use image::DynamicImage;
use ndarray::Array3;
// Use the crate name defined in Cargo.toml (sgl-model-gateway)
use smg::multimodal::vision::transforms::to_tensor as current_to_tensor;

///  manual loop implementation for comparison
fn to_tensor_legacy(image: &DynamicImage) -> Array3<f32> {
    let rgb = image.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let mut arr = Array3::<f32>::zeros((3, h, w));

    for (x, y, pixel) in rgb.enumerate_pixels() {
        let (x, y) = (x as usize, y as usize);
        arr[[0, y, x]] = pixel[0] as f32 / 255.0;
        arr[[1, y, x]] = pixel[1] as f32 / 255.0;
        arr[[2, y, x]] = pixel[2] as f32 / 255.0;
    }
    arr
}

/// The Optimized vectorized implementation
fn to_tensor_optimized(image: &DynamicImage) -> Array3<f32> {
    let rgb = image.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);

    let raw = rgb.into_raw();
    let hwc_array =
        Array3::from_shape_vec((h, w, 3), raw).expect("Buffer size should match dimensions");

    let chw_view = hwc_array.permuted_axes([2, 0, 1]);
    chw_view.mapv(|v| v as f32 / 255.0)
}

#[test]
fn test_compare_conversion_implementations() {
    // Load an existing fixture image from the repository
    let img_path = "tests/fixtures/images/large.jpg";
    let img = image::open(img_path).expect("Failed to load test image");

    println!("\n--- Image Conversion Benchmark ---");
    println!("Image Dimensions: {}x{}", img.width(), img.height());

    // 1. Correctness Check
    let output_legacy = to_tensor_legacy(&img);
    let output_optimized = to_tensor_optimized(&img);

    assert_eq!(output_legacy.shape(), output_optimized.shape());
    let diff = (&output_legacy - &output_optimized).mapv(|a| a.abs()).sum();
    assert!(diff < 1e-5, "Output values differ! Error: {}", diff);
    println!("✅ Correctness: Outputs are identical.");

    // 2. Performance Metrics
    let iterations = 100;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = to_tensor_legacy(&img);
    }
    let avg_legacy = start.elapsed() / iterations;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = to_tensor_optimized(&img);
    }
    let avg_optimized = start.elapsed() / iterations;

    println!("\nPerformance Metrics (Average over {} runs):", iterations);
    println!("- Legacy (Manual):    {:?}", avg_legacy);
    println!("- Optimized (SIMD):  {:?}", avg_optimized);

    let speedup = avg_legacy.as_secs_f64() / avg_optimized.as_secs_f64();
    println!("\n🚀 Improvement: {:.2}x faster", speedup);
}
