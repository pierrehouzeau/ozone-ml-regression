import os
import re
import base64
import hashlib
from pathlib import Path

try:
    import nbformat
except ImportError as e:
    raise SystemExit("Please install nbformat: pip install nbformat")


NB_PATH = Path('Projet_Machine_Learning.ipynb')
OUT_DIR = Path('figures/notebook')


MIME_EXT = {
    'image/png': 'png',
    'image/jpeg': 'jpg',
    'image/svg+xml': 'svg',
    'image/gif': 'gif',
}


DATA_URI_RE = re.compile(
    r'data:(?P<mime>image/(?:png|jpeg|jpg|gif|svg\+xml));base64,(?P<b64>[A-Za-z0-9+/=]+)'
)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def b64_to_bytes(s: str) -> bytes:
    if isinstance(s, list):
        s = ''.join(s)
    # Strip whitespace/newlines
    s = ''.join(str(s).split())
    return base64.b64decode(s)


def save_blob(blob: bytes, ext: str, label: str, seen: set) -> Path:
    h = hashlib.sha1(blob).hexdigest()[:10]
    if h in seen:
        return Path()
    seen.add(h)
    fname = f"{label}_{h}.{ext}"
    out_path = OUT_DIR / fname
    out_path.write_bytes(blob)
    return out_path


def save_text(text: str, ext: str, label: str, seen: set) -> Path:
    if isinstance(text, list):
        text = ''.join(text)
    blob = text.encode('utf-8')
    return save_blob(blob, ext, label, seen)


def extract_from_outputs(cell, cell_idx: int, seen: set, saved: list):
    for out_idx, out in enumerate(getattr(cell, 'outputs', []) or []):
        data = getattr(out, 'data', None) or {}
        label_base = f"cell{cell_idx:04d}_out{out_idx:02d}"
        for mime, ext in MIME_EXT.items():
            if mime in data:
                if mime == 'image/svg+xml':
                    p = save_text(data[mime], ext, label_base, seen)
                else:
                    p = save_blob(b64_to_bytes(data[mime]), ext, label_base, seen)
                if p:
                    saved.append(p)

        # Also try data URIs hidden in HTML outputs
        html = data.get('text/html') if isinstance(data, dict) else None
        if html:
            html_str = ''.join(html) if isinstance(html, list) else str(html)
            for m in DATA_URI_RE.finditer(html_str):
                mime = m.group('mime')
                b64 = m.group('b64')
                ext = MIME_EXT.get(mime, 'bin')
                p = save_blob(b64_to_bytes(b64), ext, label_base + '_html', seen)
                if p:
                    saved.append(p)


def extract_from_attachments(cell, cell_idx: int, seen: set, saved: list):
    attachments = getattr(cell, 'attachments', None) or {}
    for att_name, att in attachments.items():
        for mime, ext in MIME_EXT.items():
            if mime in att:
                label = f"cell{cell_idx:04d}_att_{att_name}"
                if mime == 'image/svg+xml':
                    p = save_text(att[mime], ext, label, seen)
                else:
                    p = save_blob(b64_to_bytes(att[mime]), ext, label, seen)
                if p:
                    saved.append(p)


def main():
    if not NB_PATH.exists():
        raise SystemExit(f"Notebook not found: {NB_PATH}")

    ensure_dir(OUT_DIR)
    nb = nbformat.read(NB_PATH, as_version=4)
    seen = set()
    saved = []
    for i, cell in enumerate(nb.cells, start=1):
        extract_from_outputs(cell, i, seen, saved)
        extract_from_attachments(cell, i, seen, saved)

    print(f"Extracted {len(saved)} images to {OUT_DIR}")
    for p in saved[:20]:
        print("-", p)
    if len(saved) > 20:
        print(f"... and {len(saved)-20} more")


if __name__ == '__main__':
    main()

