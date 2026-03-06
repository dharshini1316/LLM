cybersec-llm
============

From-scratch, GPT-style language model for cybersecurity text, plus a Streamlit “SOC Copilot” demo.

This project is intentionally small so you can train on a CPU or a modest GPU without external data or API keys.

### Features

- **From-scratch GPT-style decoder-only Transformer** in PyTorch.
- **SentencePiece BPE tokenizer** trained on a curated cybersecurity corpus.
- **Offline corpus builder** using embedded MITRE ATT&CK technique text, CVE-like descriptions, and sample log lines, with optional extra `.txt` files.
- **Training loop** with AdamW, warmup + cosine LR decay, gradient clipping, optional mixed precision, checkpointing, and resume.
- **Evaluation script** to compute perplexity.
- **Terminal generation script** and **Streamlit SOC Copilot app** for interactive analysis.

---

## Quick Start (from project root)

```powershell
cd cybersec-llm
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.data.make_corpus
python -m src.tokenizer.train_spm
python -m src.tokenizer.encode
python -m src.train.train --preset CPU_TINY --max_steps 500
$env:PYTHONPATH="."
streamlit run src/app/streamlit_app.py
```

---

## 1. Setup (Windows PowerShell)

From a PowerShell prompt in the `cybersec-llm` folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks script execution, you may need to run (as Administrator):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
```

Then open a new PowerShell window and activate the venv again.

**Python version**: Use Python **3.10+** (recommended **3.11**). Older versions (e.g. 3.7) can fail to install modern `torch`/`streamlit`.

**Important**: Run all commands below from the `cybersec-llm` project root (the folder containing `src/`, `requirements.txt`, etc.).

---

## 2. Project layout

```text
cybersec-llm/
  data/
    raw/
    cleaned/
  checkpoints/
  src/
    config.py
    data/
      make_corpus.py
      clean_text.py
    tokenizer/
      train_spm.py
      encode.py
    model/
      gpt.py
    train/
      train.py
      eval.py
    infer/
      generate.py
    app/
      streamlit_app.py
  requirements.txt
  README.md
```

---

## 3. Build the dataset

The corpus builder works entirely offline and embeds a small curated cybersecurity dataset directly in code. You can also drop additional `.txt` files into `data/raw/` to expand the dataset.

From the project root:

```powershell
python -m src.data.make_corpus
```

This will:

- Read the embedded MITRE ATT&CK, CVE-like, and log-line samples.
- Optionally read any extra `.txt` files in `data/raw/`.
- Clean and deduplicate the text.
- Write:
  - `data/cleaned/corpus.txt`
  - `data/cleaned/train.txt`
  - `data/cleaned/valid.txt`

You can safely re-run this command after adding more `.txt` files to `data/raw/`.

---

## 4. Train the tokenizer

Train a SentencePiece BPE tokenizer (default vocab size 16k):

```powershell
python -m src.tokenizer.train_spm
```

Optional arguments:

```powershell
python -m src.tokenizer.train_spm --vocab_size 32000
```

Outputs:

- `data/cleaned/spm.model`
- `data/cleaned/spm.vocab`

---

## 5. Encode the dataset

Encode `train.txt` and `valid.txt` into binary token ID arrays:

```powershell
python -m src.tokenizer.encode
```

Outputs:

- `data/cleaned/train.npy`
- `data/cleaned/valid.npy`

Each file is a 1D `uint16` NumPy array of token IDs.

---

## 6. Training the model

You can choose among three presets defined in `src/config.py`:

- **CPU_TINY** – very small model and batch size for CPU training / laptops.
- **GPU_SMALL** – small model for GPUs with around **4–6 GB** VRAM.
- **GPU_MED** – medium model for GPUs with **8–12 GB+** VRAM.

To train with the tiny CPU preset:

```powershell
python -m src.train.train --preset CPU_TINY --max_steps 500
```

Useful flags (see `python -m src.train.train --help`):

- `--preset {CPU_TINY,GPU_SMALL,GPU_MED}`
- `--max_steps` (total optimization steps)
- `--lr` (base learning rate)
- `--seed` (random seed)

Features:

- AdamW optimizer
- Linear warmup + cosine decay learning rate schedule
- Optional mixed precision with `torch.cuda.amp` when CUDA is available
- Gradient clipping
- Checkpointing to `checkpoints/` every N steps
- Resume from `checkpoints/latest.pt` with `--resume`

---

## 7. Evaluating perplexity

Compute validation perplexity from a checkpoint:

```powershell
python -m src.train.eval
```

Or with a specific checkpoint:

```powershell
python -m src.train.eval --checkpoint checkpoints/ckpt_step500.pt
```

(Omitting `--checkpoint` defaults to `checkpoints/latest.pt`.)

This will:

- Load the model config and weights from the checkpoint.
- Run the model on `data/cleaned/valid.npy`.
- Print validation loss and perplexity.

---

## 8. Text generation (terminal)

Generate text from the trained model:

```powershell
python -m src.infer.generate --checkpoint checkpoints/latest.pt --prompt "Failed SSH logins from 10.0.5.23"
```

If you omit `--prompt`, the script will prompt you interactively.

Arguments:

- `--checkpoint` (defaults to `checkpoints/latest.pt`)
- `--max_new_tokens` (default: 64)
- `--temperature` (default: 0.9)
- `--top_k` (default: 40)

---

## 9. Streamlit SOC Copilot

Run the Streamlit app (from the project root):

```powershell
$env:PYTHONPATH="."
streamlit run src/app/streamlit_app.py
```

The app will:

- Load the trained tokenizer and model checkpoint (by default `checkpoints/latest.pt`).
- Provide a text box to paste logs or security-related text.
- On **Analyze**, generate and display structured output:
  - `### Summary`
  - `### Likely Attack Type (Brute Force / SQLi / XSS / Port Scan / Malware / Suspicious Login / Unknown)`
  - `### MITRE Mapping (best guess + confidence)`
  - `### IOCs (IPs/domains/urls/hashes/ports extracted with regex)`
  - `### Defensive Actions (safe only)`
- Use simple regex-based IOC extraction and keyword-based fallback classification if the model output is unclear.
- Enforce safety: if the user asks for hacking or exploit steps, it refuses and provides defensive guidance only.

---

## 10. Hardware guidance

- **CPU-only / no GPU**
  - Use **CPU_TINY** preset.
  - Expect training to be relatively slow; keep `--max_steps` small (e.g., 500–2000).
  - Consider reducing `block_size` and `batch_size` in `src/config.py` if needed.

- **GPU with ~4–6 GB VRAM**
  - Use **GPU_SMALL** preset.
  - Keep `max_steps` modest at first (e.g., 1000–5000).

- **GPU with 8–12 GB+ VRAM**
  - Use **GPU_MED** preset.
  - You can increase `max_steps` and possibly batch size, depending on memory.

The code automatically selects CUDA when `torch.cuda.is_available()`; otherwise it runs on CPU.

---

## 11. Safety

This project is designed **only for defensive and educational cybersecurity use**. The model and the Streamlit SOC Copilot:

- Refuse to provide detailed exploit or hacking instructions.
- Focus on detection, triage, and defensive recommendations.

Use responsibly and always follow your organization’s security policies and applicable laws.

