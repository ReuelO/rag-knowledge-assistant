from dataclasses import dataclass
from pathlib import Path

DOCUMENTS_DIR = Path(__file__).parent / "data" / "documents"


@dataclass
class DocumentChunk:
    """A searchable chunk of a document."""

    content: str
    source: str
    chunk_id: int


def load_documents() -> list[DocumentChunk]:
    """Load text files and split them into paragraph-based chunks."""

    chunks = []

    for path in sorted(DOCUMENTS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        paragraphs = [
            paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()
        ]

        for chunk_id, paragraph in enumerate(paragraphs):
            chunks.append(
                DocumentChunk(
                    content=paragraph,
                    source=path.name,
                    chunk_id=chunk_id,
                )
            )

    return chunks
