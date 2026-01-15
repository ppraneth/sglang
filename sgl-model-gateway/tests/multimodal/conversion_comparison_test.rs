use std::time::Instant;

use image::DynamicImage;
use ndarray::Array3;
// Removed the unused 'current_to_tensor' import
use smg::multimodal::vision::transforms::to_tensor as to_tensor_optimized;

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

#[test]
fn test_compare_conversion_implementations() {
    let img_path = "tests/fixtures/images/large.jpg";
    let img = image::open(img_path).expect("Failed to load test image");

    let output_legacy = to_tensor_legacy(&img);
    let output_optimized = to_tensor_optimized(&img);

    let diff = (&output_legacy - &output_optimized).mapv(|a| a.abs()).sum();
    assert!(diff < 1e-5);

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

    println!(
        "\n--- Conversion Improvement: {:.2}x faster ---",
        avg_legacy.as_secs_f64() / avg_optimized.as_secs_f64()
    );
}
