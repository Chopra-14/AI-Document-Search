def create_chunks(pages, chunk_size=1000, overlap=150, min_chunk_len=60):
    """
    Splits page text into larger, semantically coherent chunks using sentence/paragraph boundaries.
    Filters out noise (like dotted table of contents lines).
    """
    chunks = []

    for page in pages:
        text = page["text"].strip()
        page_number = page["page"]

        if not text:
            continue

        # Clean repetitive dots/dashes from table of contents lines
        cleaned_lines = []
        for line in text.splitlines():
            line_str = line.strip()
            # Skip lines that are mostly dots or empty table of contents fillers
            if line_str.count(".") > 10 or line_str.count("-") > 15:
                continue
            cleaned_lines.append(line)
            
        cleaned_text = "\n".join(cleaned_lines)
        if not cleaned_text:
            cleaned_text = text

        start = 0
        text_len = len(cleaned_text)

        while start < text_len:
            end = start + chunk_size

            # If not at the end of the text, try to break at a natural boundary (newline or period)
            if end < text_len:
                boundary = max(
                    cleaned_text.rfind("\n\n", start, end),
                    cleaned_text.rfind("\n", start, end),
                    cleaned_text.rfind(". ", start, end)
                )
                if boundary != -1 and boundary > start + (chunk_size // 2):
                    end = boundary + 1

            chunk_text = cleaned_text[start:end].strip()

            if len(chunk_text) >= min_chunk_len:
                chunks.append({
                    "text": chunk_text,
                    "page": page_number
                })

            start = end - overlap if (end - overlap) > start else end

    return chunks