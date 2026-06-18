"""

Usage:
    python src/ocr/precompute_text_embeds.py
    python src/ocr/precompute_text_embeds.py \\
        --ocr_csv data/ocr_outputs/ocr_results.csv \\
        --output  data/ocr_outputs/text_embeds.pt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

CLIP_DIM = 512   # ViT-B/32 text embedding dimension
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


# ── CLIP text encoder ─────────────────────────────────────────────────────────

class CLIPTextEncoder:
    """
    Frozen CLIP ViT-B/32 text encoder.
    Encodes raw OCR text → L2-normalised 512-dim embedding.
    """

    def __init__(self):
        print(f"Loading CLIP ViT-B/32 text encoder on {DEVICE}...")
        try:
            import clip
            self.model, _ = clip.load("ViT-B/32", device=DEVICE)
            self.model.eval()
            self.backend = "openai_clip"
            print("  Loaded: OpenAI CLIP ViT-B/32")
        except Exception as e1:
            print(f"  OpenAI CLIP unavailable ({e1}), trying HuggingFace...")
            try:
                from transformers import CLIPTextModel, CLIPTokenizer
                self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
                self.model     = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
                self.model     = self.model.to(DEVICE).eval()
                self.backend   = "hf_clip"
                print("  Loaded: HuggingFace CLIP ViT-B/32")
            except Exception as e2:
                raise RuntimeError(
                    f"CLIP unavailable. Install with:\n"
                    f"  pip install git+https://github.com/openai/CLIP.git\n"
                    f"or:\n  pip install transformers\n"
                    f"OpenAI CLIP error: {e1}\nHF error: {e2}"
                )

        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_batch(self, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        """
        Encode a list of text strings → [N, 512] L2-normalised float32.
        Empty strings → zero vectors.
        """
        result = torch.zeros(len(texts), CLIP_DIM, dtype=torch.float32)

        # Group non-empty texts
        non_empty = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not non_empty:
            return result

        for batch_start in range(0, len(non_empty), batch_size):
            batch = non_empty[batch_start: batch_start + batch_size]
            idxs  = [b[0] for b in batch]
            txts  = [b[1][:250] for b in batch]   # CLIP token limit ~77 tokens

            try:
                if self.backend == "openai_clip":
                    import clip
                    tokens = clip.tokenize(txts, truncate=True).to(DEVICE)
                    embeds = self.model.encode_text(tokens)      # [B, 512]
                else:  # hf_clip
                    enc = self.tokenizer(txts, padding=True, truncation=True,
                                         max_length=77, return_tensors="pt")
                    enc = {k: v.to(DEVICE) for k, v in enc.items()}
                    out = self.model(**enc)
                    embeds = out.pooler_output                    # [B, 512]

                embeds = embeds.float()
                norms  = embeds.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                embeds = embeds / norms   # L2 normalise

                for j, idx in enumerate(idxs):
                    result[idx] = embeds[j].cpu()

            except Exception as e:
                print(f"  [WARN] Encoding batch failed: {e}")

        return result


# ── Key helpers ───────────────────────────────────────────────────────────────

def build_key(row: pd.Series) -> str:
    return (f"{row['image_path']}_"
            f"{int(float(row.get('x1', 0)))}_"
            f"{int(float(row.get('y1', 0)))}_"
            f"{int(float(row.get('x2', 0)))}_"
            f"{int(float(row.get('y2', 0)))}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr_csv",  default="data/ocr_outputs/ocr_results.csv",
                        help="OCR results CSV from extract_text.py")
    parser.add_argument("--output",   default="data/ocr_outputs/text_embeds.pt",
                        help="Output .pt file (dict[key, Tensor[512]])")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    print("Phase 4b — Precompute CLIP Text Embeddings")
    print("=" * 60)

    # ── Load OCR results ──────────────────────────────────────────────────────
    ocr_path = Path(args.ocr_csv)
    if not ocr_path.exists():
        print(f"[ERROR] OCR CSV not found: {ocr_path}")
        print("  Run: python src/ocr/extract_text.py first")
        return

    df = pd.read_csv(ocr_path)
    print(f"Loaded {len(df)} OCR records from {ocr_path}")

    # Fill empty text columns
    if "ocr_text" not in df.columns:
        df["ocr_text"] = ""
    df["ocr_text"] = df["ocr_text"].fillna("").astype(str)

    # ── Determine which records already have embeds (resume support) ──────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, torch.Tensor] = {}
    if out_path.exists():
        existing = torch.load(str(out_path), map_location="cpu")
        print(f"Resuming: {len(existing)} embeddings already cached")

    # Build index: key → (ocr_text, has_text)
    keys  = [build_key(r) for _, r in df.iterrows()]
    texts = [str(r["ocr_text"]) for _, r in df.iterrows()]
    flags = [bool(r.get("has_text", bool(t.strip()))) for r, t in
             zip([r for _, r in df.iterrows()], texts)]

    # Filter to those not yet embedded
    todo_idx  = [i for i, k in enumerate(keys) if k not in existing]
    print(f"To embed: {len(todo_idx)} new records")

    if todo_idx:
        encoder = CLIPTextEncoder()

        # Collect texts for todo batch
        todo_texts = [texts[i] if flags[i] else "" for i in todo_idx]

        # Encode in batches
        print(f"\nEncoding {len(todo_idx)} texts (batch_size={args.batch_size})...")
        all_embeds = torch.zeros(len(todo_idx), CLIP_DIM, dtype=torch.float32)

        non_empty_todo = [(j, t) for j, t in enumerate(todo_texts) if t.strip()]
        print(f"  {len(non_empty_todo)} with actual text, "
              f"{len(todo_idx) - len(non_empty_todo)} zero vectors")

        for batch_start in tqdm(range(0, len(non_empty_todo), args.batch_size),
                                desc="Encoding"):
            batch = non_empty_todo[batch_start: batch_start + args.batch_size]
            j_list = [b[0] for b in batch]
            t_list = [b[1][:250] for b in batch]

            embeds = encoder.encode_batch(t_list, batch_size=args.batch_size)
            for k, j in enumerate(j_list):
                all_embeds[j] = embeds[k]

        # Merge into existing cache
        for j, i in enumerate(todo_idx):
            existing[keys[i]] = all_embeds[j]

        torch.save(existing, str(out_path))
        print(f"\nSaved {len(existing)} embeddings → {out_path}")
    else:
        print("All embeddings already cached — nothing to do.")

    # ── Save index CSV for debugging ──────────────────────────────────────────
    index_path = out_path.with_name("embed_index.csv")
    index_rows = []
    for _, row in df.iterrows():
        key = build_key(row)
        emb = existing.get(key)
        is_zero = (emb is None or emb.norm().item() < 1e-6)
        index_rows.append({
            "key":             key,
            "image_path":      row["image_path"],
            "ocr_text":        row.get("ocr_text", ""),
            "has_text":        row.get("has_text", False),
            "text_class_prior": row.get("text_class_prior", ""),
            "embed_is_zero":   is_zero,
        })
    pd.DataFrame(index_rows).to_csv(str(index_path), index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_nonzero = sum(
        1 for k, v in existing.items() if v.norm().item() > 1e-6
    )
    print(f"\n{'='*60}")
    print("TEXT EMBED CACHE SUMMARY")
    print(f"{'='*60}")
    print(f"  Total embeddings: {len(existing)}")
    print(f"  Non-zero (text):  {n_nonzero}")
    print(f"  Zero (no text):   {len(existing) - n_nonzero}")
    print(f"  Embedding dim:    {CLIP_DIM}")
    print(f"\n  Cache file:  {out_path}  "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Index CSV:   {index_path}")
    print("\nNext: python src/classification/train_classifier.py")


if __name__ == "__main__":
    main()
