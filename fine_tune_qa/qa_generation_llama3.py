import json
import re
import subprocess
from pathlib import Path

OLLAMA_MODEL = "llama3.2:latest"
INPUT_JSONL  = Path("processed/sections_cleaned.jsonl")
OUTPUT_JSONL = Path("alpaca_qa_ollama.jsonl")

PROMPT_TMPL = """
You are a technical assistant. The following excerpt is structured under headers:

{content}

Generate ALL possible clear, specific question-and-answer pairs based on each header's content.
Ensure every answer is thorough, detailed, and directly grounded in the text. Answer should strictly be based on the input.

Format exactly as:
Q: <question>
A: <detailed answer>

Do not output anything else.
"""

QA_RE = re.compile(r"Q:\s*(.+?)\s*A:\s*(.+?)(?=\nQ:|$)", re.DOTALL)

def call_ollama(prompt: str) -> str:
    """
    Pipe `prompt` into `ollama run <MODEL>` via stdin.
    """
    proc = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt,
        text=True,
        capture_output=True
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return proc.stdout.strip()

def extract_qa(text: str):
 
    pairs = []
    for m in QA_RE.finditer(text):
        q = m.group(1).strip().replace("\n", " ")
        a = m.group(2).strip().replace("\n", " ")
        pairs.append((q, a))
    return pairs

def main():
    total_sections = total_pairs = 0

    with INPUT_JSONL.open("r", encoding="utf-8") as fin, \
         OUTPUT_JSONL.open("w", encoding="utf-8") as fout:
        
        for line in fin:
            rec     = json.loads(line)
            sec_id  = rec["id"]
            content = rec["content"]
            total_sections += 1

            prompt = PROMPT_TMPL.format(content=content)
            try:
                output = call_ollama(prompt)
            except Exception as e:
                print(f"Ollama error on {sec_id}: {e}")
                continue

            qa_list = extract_qa(output)
            if not qa_list:
                print(f"No Q&A for {sec_id}")
                continue

            for q, a in qa_list:
                fout.write(json.dumps({
                    "instruction": q,
                    "input": "",
                    "output": a
                }, ensure_ascii=False) + "\n")
                total_pairs += 1

            print(f"{len(qa_list)} Q&A from {sec_id}")

    print(f"\nDone. Processed {total_sections} sections, wrote {total_pairs} Q&A lines to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
