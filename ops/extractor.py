import io
from typing import Any, Dict, Union

import httpx
import mammoth
from markdownify import markdownify
from pypdf import PdfReader


def estimate_tokens(text: str) -> int:
    return int(len(text) / 4) + 1


async def extract_pdf(data: bytes) -> Dict[str, Any]:
    reader = PdfReader(io.BytesIO(data))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return {
        "text": text,
        "metadata": {
            "content_type": "pdf",
            "char_count": len(text),
            "estimated_tokens": estimate_tokens(text),
            "pages": len(reader.pages),
            "extraction_method": "pypdf",
        },
    }


async def extract_docx(data: bytes) -> Dict[str, Any]:
    result = mammoth.extract_raw_text(io.BytesIO(data))
    text = result.value
    return {
        "text": text,
        "metadata": {
            "content_type": "docx",
            "char_count": len(text),
            "estimated_tokens": estimate_tokens(text),
            "extraction_method": "mammoth",
        },
    }


async def extract_html(html: str) -> Dict[str, Any]:
    text = markdownify(html, heading_style="ATX", code_language="")
    return {
        "text": text,
        "metadata": {
            "content_type": "html",
            "char_count": len(text),
            "estimated_tokens": estimate_tokens(text),
            "extraction_method": "markdownify",
        },
    }


async def extract_url(url: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
    return await extract_html(response.text)


async def extract_text(content_type: str, data: Union[str, bytes]) -> Dict[str, Any]:
    lowered = content_type.lower()
    if "pdf" in lowered:
        return await extract_pdf(data if isinstance(data, bytes) else data.encode("utf-8"))
    if "docx" in lowered or lowered.endswith(".doc"):
        return await extract_docx(data if isinstance(data, bytes) else data.encode("utf-8"))
    if "html" in lowered or "htm" in lowered:
        return await extract_html(data.decode("utf-8") if isinstance(data, bytes) else data)
    if any(token in lowered for token in ["text", "txt", "markdown", "md"]):
        text = data.decode("utf-8") if isinstance(data, bytes) else data
        return {
            "text": text,
            "metadata": {
                "content_type": lowered,
                "char_count": len(text),
                "estimated_tokens": estimate_tokens(text),
                "extraction_method": "passthrough",
            },
        }
    raise ValueError(f"Unsupported content type: {content_type}")
