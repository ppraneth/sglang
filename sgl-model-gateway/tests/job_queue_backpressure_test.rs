mod common;

use std::time::Duration;

use http::{header::CONTENT_TYPE, StatusCode};
use serde_json::json;

use crate::common::test_app::TestApp;

#[tokio::test]
async fn test_job_queue_backpressure_reproduction() {
    // 1. Initialize the gateway test application
    let app = TestApp::new().await;
    let client = app.get_http_client();

    // 2. Define a worker registration payload
    // We use unique URLs to ensure they are treated as separate jobs
    let base_payload = json!({
        "url": "http://mock-worker",
        "labels": {"model": "test-model"}
    });

    println!("Starting flood of registration requests...");

    let mut accepted_count = 0;
    let mut rejected_count = 0;

    // 3. Flood the /workers endpoint with many requests
    // Currently, with an unbounded queue, all of these will return 202 Accepted.
    // Memory usage will spike as the jobs are stored in the unbounded mpsc channel.
    for i in 0..10000 {
        let mut payload = base_payload.clone();
        payload["url"] = json!(format!("http://mock-worker-{}", i));

        let response = client
            .post("/workers")
            .header(CONTENT_TYPE, "application/json")
            .body(payload.to_string())
            .send()
            .await
            .unwrap();

        match response.status() {
            StatusCode::ACCEPTED => accepted_count += 1,
            StatusCode::SERVICE_UNAVAILABLE => rejected_count += 1,
            _ => panic!("Unexpected status code: {}", response.status()),
        }

        // Periodically report progress
        if (i + 1) % 1000 == 0 {
            println!("Processed {} requests...", i + 1);
        }
    }

    println!(
        "Results: {} accepted, {} rejected (503)",
        accepted_count, rejected_count
    );

    // BUG VERIFICATION:
    // If rejected_count is 0 despite 10,000 requests, backpressure is MISSING.
    // The proposed optimization should result in rejected_count > 0
    // once the queue capacity (e.g., 100) is exceeded.
    assert!(
        rejected_count == 0,
        "If this assertion passes, it confirms the bug: the unbounded queue accepted all {} requests without backpressure.",
        accepted_count
    );
}
