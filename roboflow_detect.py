import cv2
import requests
import base64

# ------------------ CONFIG ------------------
API_KEY = "KJovHG5KYSr6kKEXbn7b"   # ⚠️ keep this safe in real projects
MODEL_ID = "led-with-hand-gesture-and-deep-learning/2"
API_URL = f"https://serverless.roboflow.com/{MODEL_ID}"

COLORS = {
    "Hand": (0, 255, 0),
    "normal": (255, 165, 0),
    "anomalie": (0, 0, 255),
}

# ------------------ HELPERS ------------------
def encode_frame(frame):
    """Convert frame to base64 string"""
    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not success:
        return None
    return base64.b64encode(buffer).decode("utf-8")


def get_predictions(frame):
    """Send frame to Roboflow API and return predictions"""
    img_b64 = encode_frame(frame)
    if img_b64 is None:
        return []

    try:
        response = requests.post(
            API_URL,
            params={"api_key": API_KEY},
            json={"image": {"type": "base64", "value": img_b64}},
            timeout=5
        )

        if response.status_code == 200:
            return response.json().get("predictions", [])
        else:
            print(f"API Error: {response.status_code}")
            return []

    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return []


def draw_predictions(frame, predictions):
    """Draw bounding boxes and labels"""
    for pred in predictions:
        x1 = int(pred["x"] - pred["width"] / 2)
        y1 = int(pred["y"] - pred["height"] / 2)
        x2 = int(pred["x"] + pred["width"] / 2)
        y2 = int(pred["y"] + pred["height"] / 2)

        label = pred["class"]
        conf = pred["confidence"]

        color = COLORS.get(label, (200, 200, 200))

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label
        cv2.putText(
            frame,
            f"{label} {conf:.0%}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return frame


# ------------------ MAIN ------------------
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not accessible")
        return

    print("▶️ Press 'Q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame not captured")
            break

        predictions = get_predictions(frame)
        frame = draw_predictions(frame, predictions)

        cv2.imshow("Roboflow Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()