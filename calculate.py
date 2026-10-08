# Back end: keeps the GloVe embeddings and does the calculations.
# The front end sends {"positive": [...], "negative": [...]}
# and gets back the closest words.

from http.server import BaseHTTPRequestHandler
import json
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Load the embeddings once, when the server starts.
# vectors.npy: 100,000 words x 50 numbers (already normalized to length 1)
VECTORS = np.load(os.path.join(DATA_DIR, "vectors.npy"))
with open(os.path.join(DATA_DIR, "words.txt"), encoding="utf-8") as f:
    WORDS = f.read().split("\n")
INDEX = {w: i for i, w in enumerate(WORDS)}


def calculate(positive, negative, topn=5):
    """Same calculation as glove.most_similar(positive=..., negative=...)."""
    missing = [w for w in positive + negative if w not in INDEX]
    if missing:
        return {"error": "Unknown word: " + ", ".join(missing)}

    # 1. find each word's embedding, 2. add the positive ones, subtract the negative ones
    result = np.zeros(VECTORS.shape[1], dtype=np.float32)
    for w in positive:
        result += VECTORS[INDEX[w]]
    for w in negative:
        result -= VECTORS[INDEX[w]]
    result /= np.linalg.norm(result)

    # 3. cosine similarity with every word, and pick the closest ones
    scores = VECTORS @ result
    used = {INDEX[w] for w in positive + negative}
    order = np.argsort(-scores)
    out = []
    for i in order:
        if int(i) in used:
            continue
        out.append({"word": WORDS[i], "similarity": round(float(scores[i]), 4)})
        if len(out) == topn:
            break
    return {"results": out}


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            positive = [str(w).strip().lower() for w in body.get("positive", []) if str(w).strip()]
            negative = [str(w).strip().lower() for w in body.get("negative", []) if str(w).strip()]
            if not positive and not negative:
                data, status = {"error": "Type at least one word."}, 400
            else:
                data = calculate(positive, negative)
                status = 400 if "error" in data else 200
        except Exception as e:
            data, status = {"error": "Server error: " + str(e)}, 500

        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
