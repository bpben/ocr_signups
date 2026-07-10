#!/usr/bin/env python3
"""Extract contact info from sign-up sheet scans using a local vision LLM.

This module provides a CLI and helper functions for OCR extraction from JPG or PDF
sign-up sheets. Configuration and prompts are loaded from external YAML files.
Optional address geocoding is supported via Nominatim.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import ollama
import pandas as pd
import yaml
from pdf2image import convert_from_path


IMAGE_SUFFIXES = {".jpg", ".jpeg"}
PDF_SUFFIXES = {".pdf"}

logger = logging.getLogger(__name__)


def setup_logging(level: str, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a"),
        ],
        force=True,
    )


def load_yaml(path: Path) -> dict:
    """Load and parse a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def load_config(config_path: Path) -> tuple[dict, str]:
    """Load project config and resolve the selected extraction prompt.

    Args:
        config_path: Path to ``config.yaml``. The prompts file is resolved relative
            to this file's parent directory.

    Returns:
        A tuple of ``(config, prompt_text)`` where ``config`` is the parsed config
        dictionary and ``prompt_text`` is the extraction prompt for ``config["prompt"]``.

    Raises:
        KeyError: If the prompt key named in config is not found in the prompts file.
    """
    config = load_yaml(config_path)
    prompts_path = config_path.parent / config.get("prompts_file", "prompts.yaml")
    prompts = load_yaml(prompts_path)
    prompt_key = config["prompt"]
    if prompt_key not in prompts:
        raise KeyError(f"Prompt '{prompt_key}' not found in {prompts_path}")
    return config, prompts[prompt_key]


def gather_inputs(path: Path) -> list[Path]:
    """Collect input files from a single path or directory.

    If ``path`` is a file, it is returned as a one-element list. If ``path`` is a
    directory, all JPG and PDF files are collected recursively.

    Args:
        path: A single scan file or a directory containing scans.

    Returns:
        Sorted list of unique input file paths.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    files = []
    for pattern in ("**/*.jpg", "**/*.jpeg", "**/*.pdf", "**/*.JPG", "**/*.JPEG", "**/*.PDF"):
        files.extend(path.glob(pattern))
    return sorted(set(files))


def pdf_to_images(pdf_path: Path, output_dir: Path) -> list[Path]:
    """Convert a PDF to JPG page images.

    Pages are saved under ``output_dir/converted/<pdf_stem>/``. Existing page
    images are reused rather than re-rendered.

    Args:
        pdf_path: Path to the source PDF.
        output_dir: Root output directory for converted images.

    Returns:
        List of paths to the converted JPG page images, in page order.
    """
    dir_name = pdf_path.stem.lower().replace(" ", "_")
    image_dir = output_dir / "converted" / dir_name
    image_dir.mkdir(parents=True, exist_ok=True)

    images = []
    pages = convert_from_path(pdf_path, 500)
    for count, page in enumerate(pages):
        image_fp = image_dir / f"out_{count}.jpg"
        if not image_fp.exists():
            page.save(image_fp, "JPEG")
        images.append(image_fp)
    return images


def resolve_images(file_path: Path, output_dir: Path) -> list[Path]:
    """Return image paths ready for OCR extraction.

    PDF inputs are converted to JPGs first. JPG inputs are returned unchanged.

    Args:
        file_path: Path to a JPG or PDF file.
        output_dir: Output directory used when converting PDFs.

    Returns:
        List of image file paths to process.

    Raises:
        ValueError: If ``file_path`` has an unsupported extension.
    """
    suffix = file_path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return pdf_to_images(file_path, output_dir)
    if suffix in IMAGE_SUFFIXES:
        return [file_path]
    raise ValueError(f"Unsupported file type: {file_path}")


def output_paths(image_fp: Path, input_root: Path, output_dir: Path) -> tuple[Path, Path]:
    """Compute output file paths for an image's raw and parsed results.

    Output paths mirror the input layout under ``output_dir``. Converted PDF pages
    are placed under ``output_dir/converted/``.

    Args:
        image_fp: Path to the source image.
        input_root: Root directory of the original input scan(s).
        output_dir: Root directory for extraction outputs.

    Returns:
        A tuple of ``(txt_path, json_path)`` for the raw model response and parsed
        JSON records.
    """
    try:
        rel = image_fp.relative_to(input_root)
    except ValueError:
        try:
            rel = Path("converted") / image_fp.relative_to(output_dir / "converted")
        except ValueError:
            rel = Path(image_fp.name)
    base = output_dir / rel.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    return base.with_suffix(".txt"), base.with_suffix(".json")


def extract_image(image_fp: Path, model: str, prompt: str) -> str:
    """Run the vision model on a single image.

    Args:
        image_fp: Path to the image file.
        model: Ollama model name (for example, ``qwen2.5vl:7b``).
        prompt: Extraction prompt sent to the model.

    Returns:
        Raw text content from the model response.
    """
    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [str(image_fp)],
        }],
        keep_alive=-1,
    )
    return response["message"]["content"]


def parse_response(response: str) -> list[dict]:
    """Parse model output into a list of record dictionaries.

    Tries to parse the full response as JSON first. If that fails, parses each
    non-empty line as a separate JSON object.

    Args:
        response: Raw model response text.

    Returns:
        List of parsed records. Returns an empty list if no valid JSON is found.
    """
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    records = []
    for line in response.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def process_image(
    image_fp: Path,
    input_root: Path,
    output_dir: Path,
    model: str,
    prompt: str,
) -> tuple[str, list[dict]] | None:
    """Extract contact info from one image and write per-image outputs.

    Skips images that already have a parsed JSON output file. On model errors,
    waits 60 seconds before returning without retrying the same image in this call.

    Args:
        image_fp: Path to the image file.
        input_root: Root directory of the original input scan(s).
        output_dir: Directory for ``.txt`` and ``.json`` outputs.
        model: Ollama model name.
        prompt: Extraction prompt sent to the model.

    Returns:
        A tuple of ``(source_path, records)`` on success, where ``source_path`` is
        the image path as a string and ``records`` is the list of parsed dicts.
        Returns ``None`` if extraction or parsing fails.
    """
    txt_fp, json_fp = output_paths(image_fp, input_root, output_dir)
    if json_fp.exists():
        logger.info("Skipping %s (already processed)", image_fp)
        with open(json_fp) as f:
            return str(image_fp), json.load(f)

    logger.info("%s", image_fp)
    try:
        response = extract_image(image_fp, model, prompt)
    except Exception as e:
        time.sleep(60)
        logger.error("Error processing %s: %s", image_fp, e, exc_info=True)
        return None

    txt_fp.write_text(response)
    records = parse_response(response)
    if not records:
        logger.warning("Failed to parse JSON for %s", image_fp)
        return None

    with open(json_fp, "w") as f:
        json.dump(records, f, indent=2)
    return str(image_fp), records


def combine_results(result_dict: dict[str, list[dict]]) -> pd.DataFrame:
    """Merge per-image extraction results into one table.

    Args:
        result_dict: Mapping from source image path to parsed record lists.

    Returns:
        Combined DataFrame with a ``source`` column identifying each image.
        Returns an empty DataFrame if no records are present.
    """
    combined = pd.DataFrame()
    for source, records in result_dict.items():
        if not records:
            continue
        df = pd.DataFrame(records)
        df["source"] = source
        combined = pd.concat([combined, df], ignore_index=True)
    return combined


def geocode_addresses(df: pd.DataFrame, geocode_cfg: dict) -> pd.DataFrame:
    """Geocode addresses and add a corrected ``fixed_address`` column.

    For each configured region, appends the region suffix to the raw address and
    queries Nominatim. The first successful match across regions is used; otherwise
    the original address is kept.

    Args:
        df: Combined extraction results. Must contain the address column named in
            ``geocode_cfg`` when geocoding is expected to run.
        geocode_cfg: Geocoding settings with optional keys:
            ``address_field`` (default: ``"Address"``), ``regions`` (list of region
            suffixes such as ``"Somerville, MA"``), and ``user_agent``.

    Returns:
        Input DataFrame with geocoding columns added when the address field exists.
        Returns the input unchanged if the address field is missing.
    """
    from geopy.geocoders import Nominatim

    address_field = geocode_cfg.get("address_field", "Address")
    if address_field not in df.columns:
        logger.warning("Geocoding skipped: column '%s' not found", address_field)
        return df

    regions = geocode_cfg.get("regions", [])
    geolocator = Nominatim(user_agent=geocode_cfg.get("user_agent", "ocr-signups"))

    def get_fixed_address(address_string: str) -> str | None:
        """Geocode one address string with Nominatim.

        Args:
            address_string: Full address query, including any region suffix.

        Returns:
            Canonical address string from Nominatim, ``"No match"`` if not found,
            or ``None`` on geocoder error.
        """
        try:
            location = geolocator.geocode(address_string, exactly_one=True)
            if location:
                return location.address
            return "No match"
        except Exception as e:
            logger.error("Geocoding error for %s: %s", address_string, e)
            return None

    fixed_by_region = {}
    for region in regions:
        col = f"fixed_{region.split(',')[0].strip().lower()}"
        df[f"address_{col}"] = df[address_field] + f", {region}"
        fixed_by_region[col] = df[f"address_{col}"].apply(get_fixed_address)

    def pick_fixed(row_idx: int) -> str:
        """Choose the best geocoded address for one row.

        Args:
            row_idx: Index of the row in ``df``.

        Returns:
            First successful geocoded address for the row, or the raw address if
            no region produced a match.
        """
        raw = df.at[row_idx, address_field]
        for col in fixed_by_region:
            fixed = fixed_by_region[col][row_idx]
            if fixed and not str(fixed).startswith("No match"):
                return fixed
        return raw

    df["fixed_address"] = [pick_fixed(i) for i in range(len(df))]
    return df


def build_final_output(df: pd.DataFrame, geocode_cfg: dict) -> pd.DataFrame:
    """Shape geocoded extraction results for CSV export.

    Args:
        df: Combined and geocoded extraction results.
        geocode_cfg: Geocoding settings; uses ``address_field`` (default:
            ``"Address"``) to locate the raw address column.

    Returns:
        DataFrame with ``name``, ``raw_address``, ``source``, and ``fixed_address``
        columns when geocoding columns are present. Otherwise returns ``df`` unchanged.
    """
    address_field = geocode_cfg.get("address_field", "Address")
    if "fixed_address" in df.columns and address_field in df.columns:
        return pd.DataFrame({
            "name": df.get("Name", df.get("name")),
            "raw_address": df[address_field],
            "source": df["source"],
            "fixed_address": df["fixed_address"],
        })
    return df


def main() -> None:
    """Run the sign-up sheet extraction CLI.

    Reads configuration from ``--config``, processes the input file or directory,
    writes per-image outputs and ``combined_output.csv``, and optionally geocodes
    addresses when enabled in config or via ``--geocode``.
    """
    parser = argparse.ArgumentParser(description="Extract sign-up sheet contact info with a vision LLM")
    parser.add_argument("input", type=Path, help="Image/PDF file or directory of scans")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml")
    geocode_group = parser.add_mutually_exclusive_group()
    geocode_group.add_argument("--geocode", action="store_true", help="Enable address geocoding")
    geocode_group.add_argument("--no-geocode", action="store_true", help="Disable address geocoding")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    config, prompt = load_config(args.config)
    output_dir = Path(config.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level, output_dir / "extract.log")

    geocode_cfg = config.get("geocode", {})
    geocode_enabled = geocode_cfg.get("enabled", False)
    if args.geocode:
        geocode_enabled = True
    elif args.no_geocode:
        geocode_enabled = False

    input_path = args.input.resolve()
    input_root = input_path if input_path.is_dir() else input_path.parent
    model = config["model"]

    result_dict: dict[str, list[dict]] = {}
    for file_path in gather_inputs(input_path):
        for image_fp in resolve_images(file_path.resolve(), output_dir):
            result = process_image(image_fp, input_root, output_dir, model, prompt)
            if result:
                result_dict[result[0]] = result[1]

    combined = combine_results(result_dict)
    if combined.empty:
        logger.warning("No observations extracted")
        return

    if geocode_enabled:
        combined = geocode_addresses(combined, geocode_cfg)
        final_output = build_final_output(combined, geocode_cfg)
    else:
        final_output = combined

    combined_csv = output_dir / "combined_output.csv"
    final_output.to_csv(combined_csv, index=False)
    logger.info("Wrote %d rows to %s", len(final_output), combined_csv)


if __name__ == "__main__":
    main()
