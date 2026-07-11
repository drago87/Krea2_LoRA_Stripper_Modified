"""
Batch strip DIT/UNET layers from Krea2 LoRAs, keeping only text-conditioning
(txtfusion) layers. Skips any file that doesn't match the Krea2 key signature.

Includes a fidelity-risk heuristic: flags files where the kept (txtfusion)
tensors make up an unusually small fraction of the original file, since those
are the LoRAs most likely to lose meaningful visual fidelity after stripping.

Features:
  * Recursively scans the given folder AND all sub-folders for .safetensors.
  * Asks UPFRONT (right after the scan path is provided) whether to move the
    original files out, and if yes, where. The destination is asked once and
    remembered for the whole run.
  * For each .safetensors that gets stripped:
      - writes <name>_stripped.safetensors
      - if <name>.jpeg (or .jpg/.png/.webp/...) exists next to the original,
        copies it to <name>_stripped.<same ext>. If absent -> silently skipped.
      - if <name>.metadata.json exists, copies it to <name>_stripped.metadata.json
        and patches file_name / file_path / preview_url / size / sha256 in place.
        If absent -> silently skipped.
      - THEN, if the user opted in, moves the ORIGINAL .safetensors + (optional)
        .jpeg + .metadata.json into the destination folder, preserving the
        relative sub-folder structure. The _stripped copies stay in place.
  * Robust against folder paths that contain spaces or quotes -- both the
    .bat launcher and this script strip quotes from user input.

Usage: python batch_strip_krea2.py "D:\\path\\to\\lora\\folder"
"""

import sys
import os
import json
import hashlib
import shutil
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

OUTPUT_SUFFIX = "_stripped"
DRY_RUN = False

# If the kept tensors are below this % of the original file's tensor bytes,
# flag the result as higher-risk for fidelity loss. Tune based on your own
# A/B testing -- this is a heuristic, not a guarantee.
RISK_THRESHOLD_PCT = 10.0

STRIP_PREFIXES_RAW = ['blocks', 'first', 'last_linear', 'last.linear', 'tmlp_', 'tproj_1']
PREFIX = 'diffusion_model.'
STRIP_PREFIXES = [PREFIX + x for x in STRIP_PREFIXES_RAW]
KREA2_SIGNATURE = 'diffusion_model.txtfusion.'

# Image extensions ComfyUI / Civitai commonly use for preview thumbnails.
# Searched in this order when looking for a preview file next to a .safetensors.
IMAGE_EXTENSIONS = ['.jpeg', '.jpg', '.png', '.webp', '.gif', '.bmp']


# ---------------------------------------------------------------------------
# Core stripping logic
# ---------------------------------------------------------------------------

def is_krea2_lora(filepath):
    try:
        with safe_open(filepath, framework="pt", device="cpu") as f:
            return any(k.startswith(KREA2_SIGNATURE) for k in f.keys())
    except Exception as e:
        print(f"  [ERROR] Could not read {filepath.name}: {e}")
        return False


def strip_layers(input_file, output_file, remove_prefixes):
    """
    Strips tensors matching remove_prefixes. Returns counts and byte sizes
    for both kept and removed tensors, so callers can compute a kept-byte
    ratio as a fidelity-risk proxy.
    """
    tensors = {}
    removed_count, kept_count = 0, 0
    removed_bytes, kept_bytes = 0, 0

    with safe_open(input_file, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        for key in f.keys():
            tensor = f.get_tensor(key)
            tensor_bytes = tensor.element_size() * tensor.nelement()

            if any(key.startswith(p) for p in remove_prefixes):
                removed_count += 1
                removed_bytes += tensor_bytes
                continue

            tensors[key] = tensor
            kept_count += 1
            kept_bytes += tensor_bytes

    save_file(tensors, output_file, metadata=metadata)
    return {
        "removed_count": removed_count,
        "kept_count": kept_count,
        "removed_bytes": removed_bytes,
        "kept_bytes": kept_bytes,
    }


# ---------------------------------------------------------------------------
# Helpers for the sidecar copy + patch step
# ---------------------------------------------------------------------------

def compute_sha256(filepath, chunk_size=1024 * 1024 * 8):
    """Stream a (potentially large) file through SHA-256 and return hex digest."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def find_preview_image(stem, directory):
    """Return the first existing image file matching <stem><ext> in `directory`,
    or None if no preview image is found."""
    for ext in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def find_associated_files(safetensors_path):
    """
    Find the .metadata.json and preview image associated with a .safetensors file.
    Returns (metadata_path, image_path) -- either may be None if not present.
    Missing sidecars are perfectly normal; callers should silently skip them.
    """
    stem = safetensors_path.stem  # filename without extension
    directory = safetensors_path.parent

    metadata_path = directory / f"{stem}.metadata.json"
    if not metadata_path.exists():
        metadata_path = None

    image_path = find_preview_image(stem, directory)
    return metadata_path, image_path


def insert_suffix_before_ext(path_str, suffix):
    """
    Insert `suffix` immediately before the file extension in a path string.
    The path separator style of the input is preserved (so a forward-slash
    Windows path like 'F:/foo/bar.safetensors' stays forward-slashed).

    Examples:
      'F:/foo/bar.safetensors' + '_stripped' -> 'F:/foo/bar_stripped.safetensors'
      'F:/foo/bar.jpeg'        + '_stripped' -> 'F:/foo/bar_stripped.jpeg'
      'bar'                    + '_stripped' -> 'bar_stripped'
      'D:\\baz\\quux.png'      + '_stripped' -> 'D:\\baz\\quux_stripped.png'
    """
    if not path_str:
        return path_str

    # Find the last path separator (either / or \).
    last_sep = max(path_str.rfind('/'), path_str.rfind('\\'))
    if last_sep >= 0:
        parent = path_str[:last_sep + 1]
        filename = path_str[last_sep + 1:]
    else:
        parent = ''
        filename = path_str

    # Insert suffix before the LAST dot in the filename portion only.
    # This correctly handles names like 'elsa_krea2-05.safetensors'
    # (the dot in 'krea2-05' is not an extension separator because there
    # is a later dot before 'safetensors').
    dot_idx = filename.rfind('.')
    if dot_idx >= 0:
        name = filename[:dot_idx]
        ext = filename[dot_idx:]
        return f"{parent}{name}{suffix}{ext}"
    return f"{parent}{filename}{suffix}"


def update_metadata_json(orig_metadata_path,
                         new_metadata_path,
                         stripped_safetensors_path):
    """
    Copy orig_metadata_path -> new_metadata_path, patching the fields that
    change when we move from the original file to its _stripped sibling:

      file_name    : append _stripped (no extension to worry about)
      file_path    : insert _stripped before .safetensors
      preview_url  : insert _stripped before the image extension
      size         : byte size of the new _stripped.safetensors file
      sha256       : SHA-256 of the new _stripped.safetensors file

    Any other fields (model_name, base_model, modified, etc.) are preserved
    unchanged from the original metadata.
    """
    with open(orig_metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_size = os.path.getsize(stripped_safetensors_path)
    new_sha256 = compute_sha256(stripped_safetensors_path)

    if 'file_name' in data and isinstance(data['file_name'], str):
        data['file_name'] = insert_suffix_before_ext(data['file_name'], OUTPUT_SUFFIX)

    if 'file_path' in data and isinstance(data['file_path'], str):
        data['file_path'] = insert_suffix_before_ext(data['file_path'], OUTPUT_SUFFIX)

    if 'preview_url' in data and isinstance(data['preview_url'], str):
        data['preview_url'] = insert_suffix_before_ext(data['preview_url'], OUTPUT_SUFFIX)

    data['size'] = new_size
    data['sha256'] = new_sha256

    with open(new_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Move-originals step (per-file)
# ---------------------------------------------------------------------------

def move_one_original(safetensors_path,
                      metadata_path,
                      image_path,
                      dest_base,
                      base_folder):
    """
    Move a single original .safetensors + (optional) .metadata.json +
    (optional) preview image into dest_base, preserving the relative
    sub-folder structure under base_folder.

    Returns (moved_safetensors_count, moved_sidecar_count, errors_list).
    Either of metadata_path / image_path may be None -- in that case the
    sidecar is silently skipped (it never existed).
    """
    moved_st = 0
    moved_sc = 0
    errors = []

    # Resolve the destination sub-folder (preserve relative path under base_folder).
    try:
        rel = safetensors_path.relative_to(base_folder)
        dest_dir = dest_base / rel.parent
    except ValueError:
        dest_dir = dest_base
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Move .safetensors
    try:
        dest_st = dest_dir / safetensors_path.name
        if dest_st.exists():
            errors.append(f"{safetensors_path}: destination already exists ({dest_st})")
        else:
            shutil.move(str(safetensors_path), str(dest_st))
            moved_st += 1
    except Exception as e:
        errors.append(f"{safetensors_path}: {e}")

    # Move .metadata.json (if exists)
    if metadata_path and metadata_path.exists():
        try:
            dest_meta = dest_dir / metadata_path.name
            if dest_meta.exists():
                errors.append(f"{metadata_path}: destination already exists ({dest_meta})")
            else:
                shutil.move(str(metadata_path), str(dest_meta))
                moved_sc += 1
        except Exception as e:
            errors.append(f"{metadata_path}: {e}")

    # Move preview image (if exists)
    if image_path and image_path.exists():
        try:
            dest_img = dest_dir / image_path.name
            if dest_img.exists():
                errors.append(f"{image_path}: destination already exists ({dest_img})")
            else:
                shutil.move(str(image_path), str(dest_img))
                moved_sc += 1
        except Exception as e:
            errors.append(f"{image_path}: {e}")

    return moved_st, moved_sc, errors


# ---------------------------------------------------------------------------
# Argument parsing robust against paths containing spaces / quotes
# ---------------------------------------------------------------------------

def parse_folder_arg():
    """
    Robustly recover the folder path from sys.argv.

    Handles three real-world cases:
      1. Properly quoted:   python script.py "D:\\path with space"
         -> argv[1] = 'D:\\path with space'  (Python already ate the quotes)
      2. Unquoted with space: python script.py D:\\path with space
         -> argv = ['script.py', 'D:\\path', 'with', 'space']
         -> we rejoin with ' '
      3. User pasted quotes that survived into argv:
         -> argv[1] = '"D:\\path with space"'  (literal quotes inside)
         -> we strip surrounding quotes
    """
    if len(sys.argv) < 2:
        return None

    raw = ' '.join(sys.argv[1:]).strip()
    # Strip a single pair of surrounding quotes (single or double).
    if len(raw) >= 2:
        if (raw[0] == '"' and raw[-1] == '"') or (raw[0] == "'" and raw[-1] == "'"):
            raw = raw[1:-1]
    return raw


def strip_surrounding_quotes(s):
    """Strip a single pair of surrounding single or double quotes from a string."""
    if len(s) >= 2:
        if (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'"):
            return s[1:-1]
    return s


def ask_move_destination(scan_folder):
    """
    Ask the user UPFRONT whether they want to move the original files out
    after each strip+copy+patch completes. If yes, ask for the destination
    folder and validate it (must exist or be creatable, must not equal the
    scan folder).

    Returns:
      Path  -> destination folder to move originals into
      None  -> user declined; originals stay in place
    """
    try:
        ans = input("Move the original files to another folder after stripping? (y/n): ").strip().lower()
    except EOFError:
        print("\nNo input received -- keeping originals in place.")
        return None

    if ans not in ('y', 'yes'):
        print("Keeping originals in place.")
        return None

    while True:
        try:
            dest = input("Enter destination folder path: ").strip()
        except EOFError:
            print("\nNo input received -- keeping originals in place.")
            return None

        dest = strip_surrounding_quotes(dest)

        if not dest:
            print("Path cannot be empty. Try again (or press Ctrl+C to abort).")
            continue

        dest_path = Path(dest)

        # Refuse to move into the scan root (would be a no-op / cause conflicts).
        try:
            if dest_path.resolve() == scan_folder.resolve():
                print("Destination must be different from the scan folder. Try again.")
                continue
        except Exception:
            pass

        try:
            dest_path.mkdir(parents=True, exist_ok=True)
            return dest_path
        except Exception as e:
            print(f"ERROR: Cannot create/use that folder: {e}")
            print("Try again (or press Ctrl+C to abort).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    raw_path = parse_folder_arg()
    if not raw_path:
        print("Usage: python batch_strip_krea2.py <folder_path>")
        sys.exit(1)

    lora_dir = Path(raw_path)

    if not lora_dir.exists() or not lora_dir.is_dir():
        print(f"ERROR: Folder not found: {lora_dir}")
        sys.exit(1)

    # Recursive scan of every .safetensors file under lora_dir.
    files = sorted(lora_dir.rglob("*.safetensors"))
    if not files:
        print(f"No .safetensors files found in {lora_dir} (or any sub-folder).")
        return

    print(f"Found {len(files)} .safetensors file(s) under {lora_dir} (recursive)")

    # ----------------------- Ask move destination UPFRONT -----------------------
    # The question is asked before any stripping happens, so the user knows
    # up-front whether originals will be relocated. The actual move happens
    # per-file, immediately after that file's _stripped outputs are written.
    print()
    print("=" * 70)
    print("MOVE ORIGINALS?")
    print("=" * 70)
    print("If yes, after each file's _stripped.safetensors (+ optional")
    print("_stripped.metadata.json + _stripped.jpeg) is written, the ORIGINAL")
    print(".safetensors (+ optional .jpeg + .metadata.json) will be moved into")
    print("the destination folder, preserving the sub-folder structure.")
    print("The _stripped copies stay where they are.")
    print()
    move_dest = ask_move_destination(lora_dir)
    if move_dest is not None:
        print(f"\nOriginals will be moved to: {move_dest}")
    print()

    # ----------------------- Process each file -----------------------
    results = []
    total_moved_st = 0
    total_moved_sc = 0
    move_errors = []

    for f in files:
        if f.stem.endswith(OUTPUT_SUFFIX):
            # Skip files that are themselves already stripped outputs.
            continue

        rel_name = str(f.relative_to(lora_dir)) if f.parent != lora_dir else f.name
        print(f"Checking: {rel_name}")

        if not is_krea2_lora(f):
            print("  -> SKIP (not Krea2 architecture / no txtfusion keys found)\n")
            results.append({"name": rel_name, "status": "skipped"})
            continue

        out_path = f.parent / f"{f.stem}{OUTPUT_SUFFIX}.safetensors"
        if out_path.exists():
            print(f"  -> SKIP (output already exists: {out_path.name})\n")
            results.append({"name": rel_name, "status": "already_done"})
            continue

        before_mb = os.path.getsize(f) / (1024 * 1024)

        if DRY_RUN:
            print(f"  -> [DRY RUN] Would strip and write {out_path.name}\n")
            results.append({"name": rel_name, "status": "dry_run", "before_mb": before_mb})
            continue

        stats = strip_layers(f, out_path, STRIP_PREFIXES)
        after_mb = os.path.getsize(out_path) / (1024 * 1024)
        reduction = 100 * (1 - after_mb / before_mb)

        # Fidelity-risk heuristic: what % of original tensor bytes survived?
        total_tensor_bytes = stats["kept_bytes"] + stats["removed_bytes"]
        kept_byte_pct = (
            100 * stats["kept_bytes"] / total_tensor_bytes
            if total_tensor_bytes > 0 else 0
        )
        is_risky = kept_byte_pct < RISK_THRESHOLD_PCT

        print(f"  -> Stripped: removed {stats['removed_count']} tensors, kept {stats['kept_count']}")
        print(f"  -> {before_mb:.2f} MB -> {after_mb:.2f} MB ({reduction:.1f}% reduction)")
        print(f"  -> Kept-byte ratio: {kept_byte_pct:.2f}%"
              f"{'  [!] LOW -- higher risk of fidelity loss, A/B test before relying on this' if is_risky else ''}")

        # ---- Copy + patch sidecar files (.jpeg preview + .metadata.json) ----
        # Missing sidecars are perfectly normal -- we silently skip them.
        metadata_path, image_path = find_associated_files(f)

        if image_path:
            new_image_path = image_path.parent / f"{f.stem}{OUTPUT_SUFFIX}{image_path.suffix}"
            try:
                shutil.copy2(str(image_path), str(new_image_path))
                print(f"  -> Copied preview image: {new_image_path.name}")
            except Exception as e:
                print(f"  -> [WARN] Could not copy preview image: {e}")
        # else: no preview image -> silently skipped

        if metadata_path:
            new_metadata_path = metadata_path.parent / f"{f.stem}{OUTPUT_SUFFIX}.metadata.json"
            try:
                update_metadata_json(metadata_path, new_metadata_path, out_path)
                print(f"  -> Wrote patched metadata: {new_metadata_path.name}")
                print(f"     size={os.path.getsize(out_path)}  sha256={compute_sha256(out_path)[:16]}...")
            except Exception as e:
                print(f"  -> [WARN] Could not patch metadata: {e}")
        # else: no metadata.json -> silently skipped

        # ---- Move the ORIGINAL files now (per-file, right after strip+copy) ----
        if move_dest is not None:
            moved_st, moved_sc, errors = move_one_original(
                safetensors_path=f,
                metadata_path=metadata_path,
                image_path=image_path,
                dest_base=move_dest,
                base_folder=lora_dir,
            )
            total_moved_st += moved_st
            total_moved_sc += moved_sc
            if errors:
                move_errors.extend(errors)
                for e in errors:
                    print(f"  -> [WARN] move error: {e}")
            else:
                side = f" (+ {moved_sc} sidecar file(s))" if moved_sc else ""
                print(f"  -> Moved original(s) to: {move_dest}{side}")

        print()

        results.append({
            "name": rel_name,
            "status": "stripped",
            "before_mb": before_mb,
            "after_mb": after_mb,
            "kept_byte_pct": kept_byte_pct,
            "risky": is_risky,
        })

    # ----------------------- Summary -----------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        if r["status"] == "stripped":
            flag = "  [!] LOW-FIDELITY RISK" if r["risky"] else ""
            print(f"[OK]      {r['name']}: {r['before_mb']:.1f}MB -> {r['after_mb']:.1f}MB "
                  f"(kept {r['kept_byte_pct']:.1f}%){flag}")
        elif r["status"] == "dry_run":
            print(f"[DRY RUN] {r['name']}: {r['before_mb']:.1f}MB (would strip)")
        elif r["status"] == "already_done":
            print(f"[SKIP]    {r['name']}: output already exists")
        else:
            print(f"[SKIP]    {r['name']}: not Krea2 architecture")

    risky_files = [r["name"] for r in results if r.get("risky")]
    if risky_files:
        print()
        print(f"NOTE: {len(risky_files)} file(s) flagged as higher-risk for fidelity loss "
              f"(kept-byte ratio under {RISK_THRESHOLD_PCT}%):")
        for name in risky_files:
            print(f"  - {name}")
        print("Recommend A/B testing these specifically before relying on the stripped version.")

    if move_dest is not None:
        print()
        print("=" * 70)
        print("MOVE SUMMARY")
        print("=" * 70)
        print(f"Destination: {move_dest}")
        print(f"Moved {total_moved_st} original .safetensors file(s) and "
              f"{total_moved_sc} sidecar file(s) (jpeg + metadata.json).")
        if move_errors:
            print(f"Errors encountered ({len(move_errors)}):")
            for e in move_errors:
                print(f"  - {e}")


if __name__ == "__main__":
    main()
