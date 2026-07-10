# Extracting sign-ups with Multimodel LLMs

This repo uses local, open-source vision LLMs (via [Ollama](https://ollama.com)) to extract structured data from scanned sign-up sheets. 

## Setup

1. **Install Ollama**

   - Mac/PC: https://ollama.com
   - Linux: `curl -fsSL https://ollama.com/install.sh | sh`

2. **Pull a vision model** (must match `model` in `config.yaml`):

   ```bash
   ollama pull qwen2.5vl:7b
   ```

3. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   If you are using PDF inputs, you will need to install Poppler to use
   `pdf2image`:
   - macOS: `brew install poppler`
   - Linux: `sudo apt install poppler-utils` (Debian/Ubuntu) or `sudo dnf install poppler-utils` (Fedora)
   - Windows: install [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases) and add its `bin` folder to `PATH`

4. **Start Ollama** before running extraction (the Ollama app or `ollama serve`).

## Using `extract.py`

The CLI reads settings from YAML and writes results under `output_dir`. See `python extract.py --help` for flags; behavior of individual functions is documented in `extract.py` docstrings.

Use `--log-level` to control verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`; default `INFO`). For example, `python extract.py scans/ --log-level ERROR` suppresses progress messages on the terminal and only shows errors. Logs are also appended to `extract.log` under `output_dir`.

### 1. Choose a prompt and model

Edit `config.yaml`:

```yaml
model: qwen2.5vl:7b
prompt: address          # key into prompts.yaml
prompts_file: prompts.yaml
output_dir: output
```

Named prompts in `prompts.yaml`:

| Key | Use for |
|-----|---------|
| `address` | Name + address columns (Somerville-style sheets) |
| `contact_form` | Structured sign-in with checkboxes (Cambridge-style) |
| `contact_freeform` | Name + email only |

Add or edit prompts to match your sheet layout. The model must return JSON (array of objects or one object per line). A prompt that describes the actual form fields usually works better than a generic one.

### 2. Run on scans

Point at a single JPG/PDF or a directory (searched recursively):

```bash
python extract.py path/to/scans/
python extract.py one_sheet.pdf
python extract.py scans/ --config my_config.yaml
```

Re-running the same inputs skips images that already have parsed JSON in `output_dir`, so you can resume interrupted batches.

### 3. Optional geocoding

For address prompts, enable geocoding in `config.yaml` or override from the CLI:

```yaml
geocode:
  enabled: true
  address_field: Address
  regions:
    - Somerville, MA
    - Cambridge, MA
  user_agent: my-address-corrector-app
```

```bash
python extract.py somerville/may_8th_2/ --geocode
python extract.py scans/ --no-geocode   # force off even if config enables it
```

When geocoding runs, `combined_output.csv` includes normalized `name`, `raw_address`, `fixed_address`, and `source` columns. Without geocoding, the CSV contains the extracted fields plus `source`.

### 4. Check outputs

Under `output_dir` (default `output/`):

- `combined_output.csv` — all rows from the run
- `extract.log` — appended run log (same messages as the terminal)
- Per-image `.txt` (raw model response) and `.json` (parsed records), laid out to mirror the input paths
- `converted/` — PDF pages rendered to JPG when needed