API
API Base URL
https://api.moondream.ai/v1/

All API requests use this base URL with skill-specific endpoints (/query, /detect, /point, /caption, /segment).

Query
Detect
Point
Caption
Segment
Visual Question Answering - Ask natural language questions about images.

curl -X POST https://api.moondream.ai/v1/query \
  -H 'Content-Type: application/json' \
  -H 'X-Moondream-Auth: YOUR_API_KEY' \
  -d '{
    "image_url": "data:image/jpeg;base64,/9j//gAQTGF2YzYxLjE5LjEwMQD/2wBDAAg+Pkk+SVVVVVVVVWRdZGhoaGRkZGRoaGhwcHCDg4NwcHBoaHBwfHyDg4+Tj4eHg4eTk5ubm7q6srLZ2eD/////xABZAAADAQEBAQAAAAAAAAAAAAAABgcFCAECAQEAAAAAAAAAAAAAAAAAAAAAEAADAAMBAQEBAAAAAAAAAAAAAQIDIREEURKBEQEAAAAAAAAAAAAAAAAAAAAA/8AAEQgAGQAZAwESAAISAAMSAP/aAAwDAQACEQMRAD8A5/PQAAABirHyVS2mUip/Pm4/vQAih9ABuRUrVLqMEALVNead7/pFgAfc+d5NLSEEAAAA/9k=",
    "question": "What is in this image?"
  }'


Response:

{
  "request_id": "2025-03-25_query_2025-03-25-21:00:39-715d03",
  "answer": "The image is a grayscale depiction of a crescent moon against a black background. The moon is rendered in varying shades of gray, appearing as a smooth, curved shape with no visible craters or details."
}


Object Detection - Identify and locate objects with bounding boxes.

curl -X POST https://api.moondream.ai/v1/detect \
  -H 'Content-Type: application/json' \
  -H 'X-Moondream-Auth: YOUR_API_KEY' \
  -d '{
    "image_url": "data:image/jpeg;base64,/9j//gAQTGF2YzYxLjE5LjEwMQD/2wBDAAg+Pkk+SVVVVVVVVWRdZGhoaGRkZGRoaGhwcHCDg4NwcHBoaHBwfHyDg4+Tj4eHg4eTk5ubm7q6srLZ2eD/////xABZAAADAQEBAQAAAAAAAAAAAAAABgcFCAECAQEAAAAAAAAAAAAAAAAAAAAAEAADAAMBAQEBAAAAAAAAAAAAAQIDIREEURKBEQEAAAAAAAAAAAAAAAAAAAAA/8AAEQgAGQAZAwESAAISAAMSAP/aAAwDAQACEQMRAD8A5/PQAAABirHyVS2mUip/Pm4/vQAih9ABuRUrVLqMEALVNead7/pFgAfc+d5NLSEEAAAA/9k=",
    "object": "moon"
  }'


Response:

{
  "request_id": "2025-03-25_detect_2025-03-25-21:00:39-715d03",
  "objects": [
    {
      "x_min": 0.2,
      "y_min": 0.3,
      "x_max": 0.6,
      "y_max": 0.8
    }
  ]
}

Object Pointing - Get precise center coordinates for objects.

curl -X POST https://api.moondream.ai/v1/point \
  -H 'Content-Type: application/json' \
  -H 'X-Moondream-Auth: YOUR_API_KEY' \
  -d '{
    "image_url": "data:image/jpeg;base64,/9j//gAQTGF2YzYxLjE5LjEwMQD/2wBDAAg+Pkk+SVVVVVVVVWRdZGhoaGRkZGRoaGhwcHCDg4NwcHBoaHBwfHyDg4+Tj4eHg4eTk5ubm7q6srLZ2eD/////xABZAAADAQEBAQAAAAAAAAAAAAAABgcFCAECAQEAAAAAAAAAAAAAAAAAAAAAEAADAAMBAQEBAAAAAAAAAAAAAQIDIREEURKBEQEAAAAAAAAAAAAAAAAAAAAA/8AAEQgAGQAZAwESAAISAAMSAP/aAAwDAQACEQMRAD8A5/PQAAABirHyVS2mUip/Pm4/vQAih9ABuRUrVLqMEALVNead7/pFgAfc+d5NLSEEAAAA/9k=",
    "object": "moon"
  }'


Response:

{
  "request_id": "2025-03-25_point_2025-03-25-21:00:39-715d03",
  "points": [
    {
      "x": 0.65,
      "y": 0.42
    }
  ]
}

Image Captioning - Generate natural language descriptions of images.

curl -X POST https://api.moondream.ai/v1/caption \
  -H 'Content-Type: application/json' \
  -H 'X-Moondream-Auth: YOUR_API_KEY' \
  -d '{
    "image_url": "data:image/jpeg;base64,/9j//gAQTGF2YzYxLjE5LjEwMQD/2wBDAAg+Pkk+SVVVVVVVVWRdZGhoaGRkZGRoaGhwcHCDg4NwcHBoaHBwfHyDg4+Tj4eHg4eTk5ubm7q6srLZ2eD/////xABZAAADAQEBAQAAAAAAAAAAAAAABgcFCAECAQEAAAAAAAAAAAAAAAAAAAAAEAADAAMBAQEBAAAAAAAAAAAAAQIDIREEURKBEQEAAAAAAAAAAAAAAAAAAAAA/8AAEQgAGQAZAwESAAISAAMSAP/aAAwDAQACEQMRAD8A5/PQAAABirHyVS2mUip/Pm4/vQAih9ABuRUrVLqMEALVNead7/pFgAfc+d5NLSEEAAAA/9k=",
    "length": "normal",
    "stream": false
  }'


Response:

{
  "caption": "A crescent moon shape is centered against a solid black background. The crescent is oriented with its convex side facing right and its concave side facing left. The image is monochromatic, with shades of gray and white. No other objects, patterns, or text are visible.",
  "metrics": {
    "input_tokens": 735,
    "output_tokens": 45,
    "prefill_time_ms": 43.51004003547132,
    "decode_time_ms": 415.3184471651912,
    "ttft_ms": 81.97528193704784
  },
  "finish_reason": "stop"
}

Image Segmentation - Generate SVG path masks for objects.

curl -X POST https://api.moondream.ai/v1/segment \
  -H 'Content-Type: application/json' \
  -H 'X-Moondream-Auth: YOUR_API_KEY' \
  -d '{
    "image_url": "data:image/jpeg;base64,...",
    "object": "cat"
  }'

Response:

{
  "path": "M 0 0.76 L 0 0.32 L 0.03 0.31 C 0.04 0.30 0.09 0.28 0.12 0.27...",
  "bbox": {
    "x_min": 0.0002,
    "y_min": 0.0039,
    "x_max": 0.7838,
    "y_max": 0.9971
  }
}