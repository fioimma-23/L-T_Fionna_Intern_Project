import os, re

input_dir  = "./markdown_output"
output_dir = "./split"

header_pattern = re.compile(r'^(#{1,6})\s+(.*)')

def slugify(text: str) -> str:
    slug = text.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"\s+", "_", slug).strip("_")
    return slug[:50] or "section"

def split_sections(lines: list[str]) -> list[dict]:
    sections, current = [], {"title": None, "body": []}
    for line in lines:
        m = header_pattern.match(line)
        if m:
            if current["title"]:
                sections.append({
                    "title": current["title"].strip(),
                    "body": "".join(current["body"]).strip()
                })
            current = {"title": line.strip(), "body": []}
        else:
            if current["title"]:
                current["body"].append(line)
    if current["title"]:
        sections.append({
            "title": current["title"].strip(),
            "body": "".join(current["body"]).strip()
        })
    return sections

print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}\n")

os.makedirs(output_dir, exist_ok=True)
file_count = 0

for root, _, files in os.walk(input_dir):
    for fname in files:
        if not fname.lower().endswith(".md"):
            continue

        file_count += 1
        src_path = os.path.join(root, fname)
        rel_root = os.path.relpath(root, input_dir)
        print(f"Processing [{file_count}]: {src_path}")

        with open(src_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        sections = split_sections(lines)
        if not sections:
            print("No headers found; skipping\n")
            continue

        dest_folder = os.path.join(output_dir, rel_root, os.path.splitext(fname)[0])
        os.makedirs(dest_folder, exist_ok=True)

        for i, sec in enumerate(sections, start=1):
            header_text = sec["title"].lstrip("#").strip()
            slug        = slugify(header_text)
            out_name    = f"{i:02d}_{slug}.md"
            out_path    = os.path.join(dest_folder, out_name)

            with open(out_path, "w", encoding="utf-8") as fo:
                fo.write(f"{sec['title']}\n\n{sec['body']}\n")
            print(f"  → Wrote: {out_path}")

print(f"\nDone. Processed {file_count} files.")
