import marker.util
marker.util.download_font = lambda *args, **kwargs: None
import os
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

input_dir = "./pdf"
output_dir = "./markdown_output"

os.makedirs(output_dir, exist_ok=True)

converter = PdfConverter(artifact_dict=create_model_dict())

for filename in os.listdir(input_dir):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(input_dir, filename)
        name = os.path.splitext(filename)[0]
        outfolder = os.path.join(output_dir, name)
        os.makedirs(outfolder, exist_ok=True)

        try:
            rendered = converter(pdf_path)
            markdown, _, _ = text_from_rendered(rendered)

            md_path = os.path.join(outfolder, f"{name}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown)

            print(f"Converted: {pdf_path} → {md_path}")
        except Exception as e:
            print(f"Failed: {pdf_path}\n{e}")
