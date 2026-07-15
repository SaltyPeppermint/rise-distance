"""Inference server for the guide ranker model.

Usage:
    uv run python server.py runs/ranker-YYYYMMDD-HHMMSS/ranker.pt
    uv run python server.py  # uses latest checkpoint

Request:
    POST /rank
    {"goal": "(some-goal)", "guides": ["(guide-1)", "(guide-2)", ...]}

Response:
    {"rankings": [
        {"guide": "(guide-1)", "predicted_class": 3, "class_name": "3", "probabilities": [...]},
        ...
    ]}
    Results are sorted by predicted class (best guides first).
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch

from ranker import find_latest_checkpoint, format_input, load_checkpoint


def make_handler(model, tokenizer, config: dict, device: torch.device):
    max_length = config["max_length"]
    class_names = config["class_names"]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/rank":
                self.send_error(404)
                return

            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))

            goal = body.get("goal", "")
            guides = body.get("guides", [])
            if not goal or not guides:
                self.send_error(400, "Need 'goal' (str) and 'guides' (list of str)")
                return

            texts = [format_input(goal, g) for g in guides]
            enc = tokenizer(
                texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                logits = model(enc["input_ids"], enc["attention_mask"])
                probs = torch.softmax(logits, dim=1).cpu().tolist()
                preds = logits.argmax(dim=1).cpu().tolist()

            rankings = [
                {
                    "guide": g,
                    "predicted_class": p,
                    "class_name": class_names[p],
                    "probabilities": prob,
                }
                for g, p, prob in zip(guides, preds, probs)
            ]
            rankings.sort(key=lambda r: r["predicted_class"])

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"rankings": rankings}).encode())

        def log_message(self, format, *args):
            print(f"[server] {args[0]}")

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Guide ranker inference server")
    parser.add_argument("checkpoint", nargs="?", type=Path, help="Path to ranker.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or find_latest_checkpoint()
    print(f"Loading checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, tokenizer = load_checkpoint(checkpoint_path, device)
    print(
        f"Model loaded ({config['model_name']}, {config['num_classes']} classes, device={device})"
    )

    handler = make_handler(model, tokenizer, config, device)
    server = HTTPServer((args.host, args.port), handler)
    print(f"Serving on http://{args.host}:{args.port}/rank")
    server.serve_forever()


if __name__ == "__main__":
    main()
