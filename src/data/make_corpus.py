from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .clean_text import clean_corpus


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


MITRE_TECHNIQUES = [
    "T1110 Brute Force: Adversaries may use brute force techniques to attempt access to accounts using passwords and password hashes.",
    "T1078 Valid Accounts: Adversaries may obtain and abuse credentials of existing accounts as a means of gaining access to systems.",
    "T1190 Exploit Public-Facing Application: Adversaries may attempt to exploit a weakness in an Internet-facing application to gain access.",
    "T1059 Command and Scripting Interpreter: Adversaries may abuse command and script interpreters to execute commands and scripts.",
    "T1047 Windows Management Instrumentation: Adversaries may use WMI to execute malicious commands or payloads.",
    "T1040 Network Sniffing: Adversaries may capture network traffic to collect credentials or other sensitive data.",
]


CVE_LIKE_DESCRIPTIONS = [
    "A vulnerability in the authentication component of the web application allows remote attackers to bypass login by sending specially crafted HTTP requests.",
    "The product fails to properly sanitize user-supplied input in SQL queries, which allows remote attackers to execute arbitrary SQL commands via crafted parameters.",
    "A stored cross-site scripting (XSS) vulnerability in the comment field allows attackers to inject arbitrary JavaScript that is executed in the context of users viewing the page.",
    "Insufficient validation of file upload parameters allows remote attackers to upload and execute arbitrary code on the server.",
    "Improper handling of JSON Web Tokens (JWT) allows attackers to forge tokens and gain unauthorized access to protected endpoints.",
    "The SSH service is configured to allow password authentication with weak credentials, making it susceptible to brute-force attacks.",
]


SAMPLE_LOG_LINES = [
    # Auth logs
    "2024-10-12T03:12:44Z sshd[1023]: Failed password for invalid user admin from 203.0.113.45 port 55232 ssh2",
    "2024-10-12T03:12:47Z sshd[1023]: Failed password for invalid user admin from 203.0.113.45 port 55232 ssh2",
    "2024-10-12T03:12:50Z sshd[1023]: Failed password for invalid user admin from 203.0.113.45 port 55232 ssh2",
    "2024-10-12T03:13:01Z sshd[1120]: Accepted password for alice from 198.51.100.20 port 60212 ssh2",
    "2024-10-12T04:02:10Z sshd[1301]: Failed password for root from 192.0.2.15 port 43022 ssh2",
    # Web logs
    '192.0.2.10 - - [12/Oct/2024:04:15:01 +0000] "GET /login.php HTTP/1.1" 200 512 "-" "Mozilla/5.0"',
    '192.0.2.10 - - [12/Oct/2024:04:15:02 +0000] "POST /login.php HTTP/1.1" 401 256 "-" "Mozilla/5.0"',
    '203.0.113.77 - - [12/Oct/2024:04:15:10 +0000] "GET /index.php?id=1%20UNION%20SELECT%201,2,3 HTTP/1.1" 500 1024 "-" "sqlmap"',
    '203.0.113.77 - - [12/Oct/2024:04:16:21 +0000] "GET /search.php?q=<script>alert(1)</script> HTTP/1.1" 200 768 "-" "curl/8.0"',
    # Firewall / port scan
    "2024-10-12T05:00:01Z firewall: DENY TCP 203.0.113.99:54321 -> 10.0.0.5:22 SYN",
    "2024-10-12T05:00:02Z firewall: DENY TCP 203.0.113.99:54321 -> 10.0.0.5:23 SYN",
    "2024-10-12T05:00:03Z firewall: DENY TCP 203.0.113.99:54321 -> 10.0.0.5:25 SYN",
    # Malware / suspicious
    "Endpoint AV detected Trojan.Generic on host WIN-10-CLIENT in file C:\\Users\\Public\\Downloads\\invoice.exe",
    "EDR alert: Suspicious PowerShell command invoking encoded payload from hxxp://malicious.example.com/a.ps1",
    "User bob logged in from unusual geographic location; impossible travel detected between logins.",
]


def load_extra_raw_files(raw_dir: Path) -> List[str]:
    texts: List[str] = []
    if not raw_dir.exists():
        return texts
    for path in sorted(raw_dir.glob("*.txt")):
        try:
            texts.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return texts


def build_corpus() -> None:
    root = get_project_root()
    raw_dir = root / "data" / "raw"
    cleaned_dir = root / "data" / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    texts: List[str] = []
    texts.extend(MITRE_TECHNIQUES)
    texts.extend(CVE_LIKE_DESCRIPTIONS)
    texts.extend(SAMPLE_LOG_LINES)
    texts.extend(load_extra_raw_files(raw_dir))

    # Split into individual lines for cleaning.
    raw_lines: List[str] = []
    for t in texts:
        raw_lines.extend(t.splitlines())

    cleaned_lines = clean_corpus(raw_lines)

    if not cleaned_lines:
        raise SystemExit("No cleaned lines produced; add some raw .txt files to data/raw and try again.")

    corpus_path = cleaned_dir / "corpus.txt"
    train_path = cleaned_dir / "train.txt"
    valid_path = cleaned_dir / "valid.txt"

    corpus_text = "\n".join(cleaned_lines)
    corpus_path.write_text(corpus_text, encoding="utf-8")

    # Simple line-based split into train/valid (90/10).
    n = len(cleaned_lines)
    split = max(1, int(n * 0.9))
    train_lines = cleaned_lines[:split]
    valid_lines = cleaned_lines[split:] or cleaned_lines[-max(1, n // 10) :]

    train_path.write_text("\n".join(train_lines), encoding="utf-8")
    valid_path.write_text("\n".join(valid_lines), encoding="utf-8")

    print(f"Wrote corpus with {len(cleaned_lines)} lines")
    print(f"  corpus.txt: {corpus_path}")
    print(f"  train.txt:  {train_path}")
    print(f"  valid.txt:  {valid_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build small cybersecurity corpus.")
    parser.parse_args()  # No options for now, but keeps interface extensible.
    build_corpus()


if __name__ == "__main__":
    main()

