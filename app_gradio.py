import json
import tempfile
from pathlib import Path

import gradio as gr

from skin_lesion_attention.predict import predict


DEFAULT_CHECKPOINT = "outputs/checkpoints/best.pt"


def run_prediction(image, checkpoint_path):
    if image is None:
        return "Please upload an image.", [], "{}"

    checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_image_path = tmp.name
        image.save(temp_image_path)

    try:
        result = predict(checkpoint_path, temp_image_path)
    finally:
        Path(temp_image_path).unlink(missing_ok=True)

    predicted_class = result.get("predicted_class", "unknown")
    probability = result.get("probabilities", {}).get(predicted_class)

    if isinstance(probability, (int, float)):
        summary = f"Predicted class: {predicted_class} ({probability:.2%})"
    else:
        summary = f"Predicted class: {predicted_class}"

    probability_rows = sorted(
        result.get("probabilities", {}).items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return summary, probability_rows, json.dumps(result, indent=2)


def build_app():
    with gr.Blocks(title="Skin Lesion Predictor") as demo:
        gr.Markdown("# Skin Lesion Predictor")

        checkpoint = gr.Textbox(
            label="Checkpoint path",
            value=DEFAULT_CHECKPOINT,
        )
        image = gr.Image(type="pil", label="Upload lesion image")
        predict_button = gr.Button("Predict", variant="primary")

        summary = gr.Textbox(label="Summary", interactive=False)
        probabilities = gr.Dataframe(
            headers=["Class", "Probability"],
            label="Class probabilities",
            interactive=False,
        )
        result_json = gr.Code(label="Full JSON result", language="json")

        predict_button.click(
            fn=run_prediction,
            inputs=[image, checkpoint],
            outputs=[summary, probabilities, result_json],
        )

    return demo


if __name__ == "__main__":
    build_app().launch(share=True)
