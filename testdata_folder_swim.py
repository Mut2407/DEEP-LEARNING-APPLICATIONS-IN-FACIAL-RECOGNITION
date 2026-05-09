import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import timm
import matplotlib.pyplot as plt
import numpy as np

# --- Config ---
MODEL_PATH = "swin_best_model.pth"
INPUT_FOLDER = "test10/surprise"
OUTPUT_FOLDER = "results/surprise"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(CLASS_NAMES)
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


# --- Load model ---
def load_model(model_path, num_classes):
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False)
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    model = model.to(DEVICE)
    model.eval()
    return model


# --- PREPROCESS ---
def get_test_transform(img_size):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


# --- Predict & Save output image ---
def predict_and_save(model, image_path, transform, class_names, device):
    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)  # output shape: (1, 7, 7, 7)

    # Global average pooling
    output = output.mean(dim=[2, 3])  # shape -> (1, 7)

        # Logits shape: (7)
    logits = output[0]
    probabilities = torch.softmax(logits, dim=0)

    predicted_idx = probabilities.argmax().item()
    predicted_class = class_names[predicted_idx]
    confidence = probabilities[predicted_idx].item()

    # === LOSS ===
    criterion = nn.CrossEntropyLoss()
    label_idx = class_names.index("surprise") 
    label_tensor = torch.tensor([label_idx]).to(device)

    loss = criterion(logits.unsqueeze(0), label_tensor)

    print(f"\n--- Result: {image_path} ---")
    print(f"Emotion: {predicted_class}")
    print(f"Accuracy: {confidence * 100:.2f}%")
    print(f"Loss: {loss.item():.4f}")


    # Create text label
    scale = 1.5
    new_image_size = (int(image.width * scale), int(image.height * scale))
    image = image.resize(new_image_size, Image.LANCZOS)
    label = f"{predicted_class} ({confidence:.3f})"

    # Font
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    # Calculate text size
    dummy = Image.new("RGB", (1, 1))
    draw_dummy = ImageDraw.Draw(dummy)
    bbox = draw_dummy.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    padding = 20
    new_width = max(image.width, text_w + padding * 2)
    new_height = text_h + padding * 2 + image.height

    # Create new canvas (white background)
    final_img = Image.new("RGB", (new_width, new_height), (255, 255, 255))
    draw = ImageDraw.Draw(final_img)

    # Draw text at the top center
    text_x = (new_width - text_w) // 2
    text_y = padding
    draw.text((text_x, text_y), label, font=font, fill=(0, 0, 0))

    # Paste original image below the text
    final_img.paste(image, (0, text_h + padding * 2))

    # Save output
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(image_path))
    final_img.save(output_path)

    print(f"Saved: {output_path}")


# --- MAIN ---
if __name__ == "__main__":
    if not os.path.isdir(INPUT_FOLDER):
        print(f"Lỗi: Folder '{INPUT_FOLDER}' không tồn tại.")
        sys.exit(1)

    model = load_model(MODEL_PATH, NUM_CLASSES)
    transform = get_test_transform(IMG_SIZE)

    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith(VALID_EXTENSIONS):
            path = os.path.join(INPUT_FOLDER, filename)
            predict_and_save(model, path, transform, CLASS_NAMES, DEVICE)
