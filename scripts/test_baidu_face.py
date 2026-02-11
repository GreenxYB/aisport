import argparse
import base64
import os

import cv2

try:
    from aip import AipFace  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("baidu-aip not installed. pip install baidu-aip") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Baidu AIP face search")
    parser.add_argument("--image", required=True, help="Path to face image")
    parser.add_argument("--group", default=os.getenv("BAIDU_GROUP_ID", "default"))
    args = parser.parse_args()

    app_id = '120850173'
    api_key = 'GcJ4mfJp7GSGJ9fvReCP9mcQ'
    secret_key = 'QzaVHxXZnU50BufoFpYiVWzjnJbJW1O7'
    if not (app_id and api_key and secret_key):
        raise SystemExit("Missing BAIDU_APP_ID / BAIDU_API_KEY / BAIDU_SECRET_KEY")

    client = AipFace(app_id, api_key, secret_key)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".jpg", rgb)
    if not ok:
        raise SystemExit("Failed to encode image to jpg")

    face_b64 = base64.b64encode(buf).decode("utf-8")
    res = client.search(face_b64, "BASE64", args.group)
    print(res)


if __name__ == "__main__":
    main()
