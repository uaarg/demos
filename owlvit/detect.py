import argparse
from PIL import Image, ImageDraw
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection


def load_image(image_source: str) -> Image.Image:
    """Load image from a URL or local path."""
    return Image.open(image_source).convert("RGB")


def draw_boxes(image: Image.Image, boxes, labels, scores, texts):
    """Draw bounding boxes with labels and scores on an image."""
    draw = ImageDraw.Draw(image)
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        text_label = f"{texts[label]} ({score.item():.2f})"
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0] + 4, box[1] + 4), text_label, fill="red")
    return image


def main():
    parser = argparse.ArgumentParser(description="Run OwlViT object detection and save image with bounding boxes.")
    parser.add_argument("--image", required=True, help="Path or URL of the image.")
    parser.add_argument("--output", required=True, help="Output path to save the image with boxes.")
    parser.add_argument("--texts", nargs="+", default=["a photo of a cat", "a photo of a dog"],
                        help="Text queries for object detection.")
    parser.add_argument("--threshold", type=float, default=0.1, help="Detection confidence threshold.")
    args = parser.parse_args()

    # Load models
    model_name = "google/owlv2-base-patch16"
    print(f"Loading OWL-ViT model ({model_name})...")
    processor = Owlv2Processor.from_pretrained(model_name)
    model = Owlv2ForObjectDetection.from_pretrained(model_name)

    # Load image
    image = load_image(args.image)
    texts = [args.texts]

    # Run model
    import time
    start = time.time()
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Post-process results
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs,
                                                      threshold=args.threshold,
                                                      target_sizes=target_sizes)
    end = time.time()
    boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

    # Draw and save
    annotated = draw_boxes(image, boxes, labels, scores, texts[0])
    annotated.save(args.output)
    print(f"Saved output to {args.output}")

    # Print results
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {texts[0][label]} with confidence {round(score.item(), 3)} at location {box}")

    print(f"Took {end - start:.2f} seconds to run inference.")


if __name__ == "__main__":
    main()
