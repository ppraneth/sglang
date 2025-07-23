import torch

import sglang as sgl


def main():
    """
    Integration test for loading and running a model quantized with the
    FBGEMM FP8 scheme using SGLang.
    """
    # This requires a CUDA-enabled GPU with sufficient VRAM (~20GB recommended for 8B model)
    if not torch.cuda.is_available():
        print("CUDA is not available. This test requires a CUDA-enabled GPU.")
        return

    # The path to the pre-quantized model on the Hugging Face Hub
    model_path = "nm-testing/Meta-Llama-3-8B-Instruct-FBGEMM-nonuniform"

    print(f"Loading model: {model_path}")
    print("This will test the entire FBGEMMFp8 quantization pipeline...")

    # --- Crucial Part ---
    # Instantiate the SGLang LLM engine.
    # We specify quantization="fbgemm_fp8" to tell the engine to activate
    # the loading logic for this quantization method.
    try:
        engine = sgl.LLM(
            model_path=model_path,
            quantization="fbgemm_fp8",
            tensor_parallel_size=1,  # Use 1 GPU for this test
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(
            f"!!! Model loading failed. This indicates a problem in the quantization pipeline."
        )
        raise e

    # Define a simple generation task to see if the model works
    @sgl.function
    def generate_poem(s, topic: str):
        # Llama-3-Instruct uses a specific chat template format
        s += sgl.user(f"Write a short, four-line poem about {topic}.")
        s += sgl.assistant(sgl.gen("response", max_tokens=128, stop="<|eot_id|>"))

    print("\nRunning a generation task...")
    topic = "the ocean"
    state = generate_poem.run(topic=topic)

    response_text = state["response"].strip()

    print("-" * 20)
    print(f"Prompt: Write a short, four-line poem about {topic}.")
    print(f"Generated Response:\n{response_text}")
    print("-" * 20)

    # --- Verification ---
    # A successful test means the model ran and produced coherent text.
    assert len(response_text) > 20, "Generated response is too short or empty."
    # A simple check to ensure it's not just repeating the same character
    assert (
        len(set(response_text.lower())) > 5
    ), "Generated response lacks variety, possibly garbage."

    print("\n✅ Test Passed: The model loaded and generated coherent text.")


if __name__ == "__main__":
    main()
