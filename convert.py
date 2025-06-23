import json
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants -----------------------------------------------------------------
# ---------------------------------------------------------------------------

# Exact block found in many *description* fields that we want to shrink
OLD_SNIPPET_DESCRIPTION = (
    "\n\nGeneric Pattern of the input numerical values: \n\n"
    "Series 1:\n\nx_1\n\nx_2\n\n...\n\nx_N\n\n\n\n"
    "Series 2:\n\ny_1\n\ny_2\n\n...\n\ny_N\n\n**"
)

# Replacement for both *description* and *question* (input)
NEW_SNIPPET_COMPACT = (
    "\n\nGeneric Pattern of the input numerical values: "
    "Series: [Series1: [x_1, x_2, ..., x_n], Series2: [y_1, y_2, ..., y_n]]\n\n"
)

# Regex that catches the verbose block inside the *description*
GENERIC_PATTERN_RE_DESCRIPTION = re.compile(
    r"Generic Pattern of the input numerical values:[\s\S]*?\*\*",
    re.MULTILINE,
)

# Regex that catches the verbose block in the *question*/instruction text.
# It stops right before the (**Instruction:**) header so we preserve it.
GENERIC_PATTERN_RE_INPUT = re.compile(
    r"Generic Pattern of the input numerical values:[\s\S]*?\n\n(?=\*\*Instruction:)",
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Utility helpers ------------------------------------------------------------
# ---------------------------------------------------------------------------

def _series_to_str(series: list[Any]) -> str:
    """Return a human‑readable string representation of the *series* field."""
    if (
        isinstance(series, list)
        and len(series) == 2
        and all(isinstance(sub, list) for sub in series)
    ):
        s1 = ", ".join(f"{x:02d}" for x in series[0])
        s2 = ", ".join(f"{x:02d}" for x in series[1])
        return f"Series1: [{s1}], Series2: [{s2}]"
    # Fallback (flat list or something else)
    return ", ".join(f"{x:02d}" for x in series)


def _to_printable(value: Any) -> str:
    """Force *value* to a (unicode) string so we can run regex on it."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _compact_generic_block(text: str, is_description: bool) -> str:
    """Replace the verbose placeholder block with its compact form."""
    if is_description:
        if OLD_SNIPPET_DESCRIPTION in text:
            return text.replace(OLD_SNIPPET_DESCRIPTION, NEW_SNIPPET_COMPACT)
        return GENERIC_PATTERN_RE_DESCRIPTION.sub(NEW_SNIPPET_COMPACT, text)
    # For *input* / question side:
    return GENERIC_PATTERN_RE_INPUT.sub(NEW_SNIPPET_COMPACT, text)

# ---------------------------------------------------------------------------
# Main conversion routine ----------------------------------------------------
# ---------------------------------------------------------------------------

def convert_dataset_to_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    truncate: int = 1000,
) -> None:
    """Convert a JSONL dataset to the *{index, input, output}* JSONL format.

    Both *description* **and** *question* blocks get the verbose numbered‑list
    placeholder replaced with the new compact version.
    """

    input_path, output_path = Path(input_path), Path(output_path)

    # ----------------------------------------------------------------------
    # Load the source dataset (truncate if requested) -----------------------
    # ----------------------------------------------------------------------
    with input_path.open("r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f][:truncate]

    with output_path.open("w", encoding="utf-8") as out_file:
        print("Found", len(data), "examples")

        for item in data:
            # --------------------------------------------------------------
            # Source fields -------------------------------------------------
            # --------------------------------------------------------------
            index: int | None = item.get("index")
            raw_question = _to_printable(item.get("question", ""))
            raw_description = _to_printable(item.get("description", ""))
            series = item.get("series", [])

            # --------------------------------------------------------------
            # Compact the generic‑pattern block in *both* fields ------------
            # --------------------------------------------------------------
            question = _compact_generic_block(raw_question, is_description=False)
            description = _compact_generic_block(raw_description, is_description=True)

            # --------------------------------------------------------------
            # Build the *input* string -------------------------------------
            # --------------------------------------------------------------
            series_str = _series_to_str(series)
            input_text = f"{question} Series: [{series_str}]"

            # --------------------------------------------------------------
            # Emit transformed JSONL line ----------------------------------
            # --------------------------------------------------------------
            jsonl_obj = {
                "index": index,
                "input": input_text,
                "output": description,
            }

            out_file.write(json.dumps(jsonl_obj, ensure_ascii=False) + "\n")

    print(f"Conversion terminée : {output_path}")

# ---------------------------------------------------------------------------
# Example CLI entry‑point ----------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    convert_dataset_to_jsonl(
        "/home/u2/pawlus/dataset/data2.jsonl",
        "/home/u2/pawlus/dataset/test_jsonl.jsonl",
        truncate=1000,
    )
