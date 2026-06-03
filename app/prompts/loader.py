"""Load Gemini prompt templates from app/prompts/."""

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent
ANALYSIS_PROMPT = "analysis.txt"

_VALIDATION_WITH_USER = """
10. VALIDATION (user provided asset name — required):
   - User's Asset Name: {user_asset_name}
   - User's Description: {user_description}

   Validate whether the user's name/description matches the asset in the images. Rules:
   - Generic category names (Chair, Laptop, Pump) without brand → APPROVE unless image shows completely different object
   - Model names without brand → APPROVE if image matches that model's parent brand
   - Contradicting brands (user says Dell, image shows HP logo) → REJECT
   - Allow typos and phonetic variants (Lenevo = Lenovo)
   - Do NOT confuse barcode sticker logos with manufacturer brands

   Also return:
   - "namedescriptionmatch": "Y" or "N"
   - "namedescriptionmatchpercent": 0-100
   - "reasoning": "User Claim: [Type]. Image Shows: [Name]. Verdict: [Explanation]."
"""

_VALIDATION_OMIT = (
    "Do NOT include namedescriptionmatch, namedescriptionmatchpercent, or reasoning fields. "
    "Omit them or set them to null."
)


@lru_cache
def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def get_analysis_prompt(
    user_asset_name: str | None = None,
    user_description: str | None = None,
) -> str:
    template = load_prompt(ANALYSIS_PROMPT)
    if user_asset_name and user_asset_name.strip():
        validation = _VALIDATION_WITH_USER.format(
            user_asset_name=user_asset_name.strip(),
            user_description=(
                user_description.strip()
                if user_description
                else "(empty - validation based on name only)"
            ),
        )
    else:
        validation = _VALIDATION_OMIT
    return template.format(validation_section=validation)


def get_composite_analysis_prompt(
    user_asset_name: str | None = None,
    user_description: str | None = None,
) -> str:
    """Alias for get_analysis_prompt (backward compatibility)."""
    return get_analysis_prompt(user_asset_name, user_description)
