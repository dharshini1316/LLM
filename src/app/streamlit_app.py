from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import sentencepiece as spm
import streamlit as st
import torch

from src.model.gpt import GPT, GPTConfig


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model_and_tokenizer() -> Tuple[GPT, spm.SentencePieceProcessor, torch.device]:
    root = get_project_root()
    ckpt_path = root / "checkpoints" / "latest.pt"
    spm_path = root / "data" / "cleaned" / "spm.model"

    device = get_device()
    if not ckpt_path.exists():
        st.warning("Checkpoint not found at checkpoints/latest.pt. Train the model first.")
        st.stop()
    if not spm_path.exists():
        st.warning("SentencePiece model not found at data/cleaned/spm.model. Train the tokenizer and encode first.")
        st.stop()

    # PyTorch 2.6+ defaults to weights_only=True; this checkpoint also stores
    # configuration and optimizer state, so we explicitly allow loading them.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig.from_dict(ckpt["model_config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sp = spm.SentencePieceProcessor()
    sp.load(str(spm_path))

    return model, sp, device


def generate_text(
    model: GPT,
    sp: spm.SentencePieceProcessor,
    device: torch.device,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.9,
    top_k: int = 40,
) -> str:
    ids = sp.encode(prompt, out_type=int)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    out_ids = y[0].tolist()
    return sp.decode(out_ids)


IOC_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
IOC_URL_RE = re.compile(r"\bhttps?://[^\s]+", re.IGNORECASE)
IOC_DOMAIN_RE = re.compile(r"\b([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b")
IOC_HASH_RE = re.compile(r"\b[a-fA-F0-9]{32,64}\b")
IOC_PORT_RE = re.compile(r"\bport\s+(\d{1,5})\b", re.IGNORECASE)


def extract_iocs(text: str) -> Dict[str, List[str]]:
    ips = sorted(set(IOC_IP_RE.findall(text)))
    urls = sorted(set(IOC_URL_RE.findall(text)))
    domains = sorted(
        {d for d in IOC_DOMAIN_RE.findall(text) if not d.lower().endswith(("localhost", "local"))}
    )
    hashes = sorted(set(IOC_HASH_RE.findall(text)))
    ports = sorted(set(IOC_PORT_RE.findall(text)))
    return {
        "ips": ips,
        "urls": urls,
        "domains": domains,
        "hashes": hashes,
        "ports": ports,
    }


ATTACK_LABELS = [
    "Brute Force",
    "SQLi",
    "XSS",
    "Port Scan",
    "Malware",
    "Suspicious Login",
    "Unknown",
]


def keyword_fallback_attack_type(text: str) -> str:
    t = text.lower()
    if "failed password" in t or "invalid user" in t or "authentication failure" in t:
        return "Brute Force"
    if "union select" in t or "sqlmap" in t or "sql injection" in t:
        return "SQLi"
    if "<script>" in t or "xss" in t:
        return "XSS"
    if "port scan" in t or "nmap" in t or "syn" in t and "firewall" in t:
        return "Port Scan"
    if "malware" in t or "trojan" in t or "ransomware" in t or "suspicious powershell" in t:
        return "Malware"
    if "impossible travel" in t or "unusual geographic" in t or "suspicious login" in t:
        return "Suspicious Login"
    return "Unknown"


def map_attack_to_mitre(label: str) -> str:
    mapping = {
        "Brute Force": "T1110 Brute Force (High confidence)",
        "SQLi": "T1190 Exploit Public-Facing Application (Medium confidence)",
        "XSS": "T1190 Exploit Public-Facing Application (Medium confidence)",
        "Port Scan": "T1046 Network Service Discovery (Medium confidence)",
        "Malware": "T1204 User Execution / T1059 Command and Scripting Interpreter (Low–Medium confidence)",
        "Suspicious Login": "T1078 Valid Accounts (Medium confidence)",
        "Unknown": "Unable to confidently map to a specific technique.",
    }
    return mapping.get(label, "Unable to confidently map to a specific technique.")


def heuristic_summary(text: str, attack_label: str, iocs: Dict[str, List[str]]) -> str:
    parts: List[str] = []
    if attack_label != "Unknown":
        parts.append(f"Likely {attack_label} activity based on log keywords/patterns.")
    else:
        parts.append("Potentially security-relevant activity detected; classification is uncertain.")

    if iocs.get("ips"):
        parts.append(f"Observed source/destination IPs: {', '.join(iocs['ips'][:6])}{'…' if len(iocs['ips']) > 6 else ''}.")
    if iocs.get("urls"):
        parts.append(f"Observed URLs: {', '.join(iocs['urls'][:3])}{'…' if len(iocs['urls']) > 3 else ''}.")
    if iocs.get("domains"):
        parts.append(f"Observed domains: {', '.join(iocs['domains'][:3])}{'…' if len(iocs['domains']) > 3 else ''}.")
    if iocs.get("hashes"):
        parts.append(f"Observed hashes: {', '.join(iocs['hashes'][:2])}{'…' if len(iocs['hashes']) > 2 else ''}.")
    if iocs.get("ports"):
        parts.append(f"Observed ports: {', '.join(iocs['ports'][:8])}{'…' if len(iocs['ports']) > 8 else ''}.")

    t = text.lower()
    if "failed password" in t:
        parts.append("Multiple authentication failures suggest credential guessing or password spraying.")
    if "union select" in t or "sqlmap" in t:
        parts.append("Query patterns suggest automated probing for SQL injection.")
    if "<script>" in t:
        parts.append("Request content includes script tags consistent with XSS probing.")
    if "firewall" in t and "deny" in t and "syn" in t:
        parts.append("Repeated blocked SYN attempts across ports suggest scanning behavior.")
    if "powershell" in t and ("encoded" in t or "base64" in t):
        parts.append("PowerShell indicators may suggest suspicious script execution.")

    return " ".join(parts)


def defensive_actions_template(attack_label: str) -> str:
    if attack_label == "Brute Force":
        return (
            "- Check authentication logs for the affected accounts and source IPs.\n"
            "- Enforce MFA and strong password policies; consider temporary lockouts/rate limiting.\n"
            "- Block or throttle offending IPs at the firewall/VPN/IdP.\n"
            "- Review successful logins following the failures for potential compromise.\n"
            "- If compromise suspected, reset credentials and rotate keys/tokens."
        )
    if attack_label == "SQLi":
        return (
            "- Review WAF/web logs for injection payloads and affected endpoints.\n"
            "- Patch/validate input handling; use parameterized queries and ORM protections.\n"
            "- Add WAF rules for common SQLi patterns and monitor for bypass attempts.\n"
            "- Check database audit logs for anomalous queries and data access.\n"
            "- Rotate DB credentials if exposure is suspected."
        )
    if attack_label == "XSS":
        return (
            "- Identify the vulnerable parameter/page and confirm output encoding.\n"
            "- Apply output encoding and a strong Content Security Policy (CSP).\n"
            "- Add WAF rules to detect script-tag payloads; monitor for persistence.\n"
            "- Review user sessions for suspicious activity and force logout if needed.\n"
            "- Validate that stored content is sanitized server-side."
        )
    if attack_label == "Port Scan":
        return (
            "- Confirm whether the source IP is authorized (scanner, monitoring, pentest).\n"
            "- Block or rate-limit scanning sources at perimeter controls.\n"
            "- Verify exposed services, patch levels, and disable unnecessary ports.\n"
            "- Add IDS/IPS detections for scan patterns; watch for follow-on exploitation.\n"
            "- Review asset inventory and firewall rules for least exposure."
        )
    if attack_label == "Malware":
        return (
            "- Isolate affected endpoints and collect EDR telemetry.\n"
            "- Quarantine the file/process and run a full scan; capture hashes and parent process.\n"
            "- Inspect persistence mechanisms (scheduled tasks, registry run keys, services).\n"
            "- Search across fleet for the same indicators (hash, domain, URL, command line).\n"
            "- If confirmed, follow incident response playbooks and restore from known-good backups."
        )
    if attack_label == "Suspicious Login":
        return (
            "- Validate the login is expected (user confirmation, device posture, geo/ASN checks).\n"
            "- Enforce MFA and review conditional access policies.\n"
            "- Rotate credentials if anomalous session behavior is observed.\n"
            "- Review subsequent actions (accessed resources, privilege changes, new tokens).\n"
            "- Add monitoring for impossible travel and unusual device fingerprints."
        )
    return (
        "- Increase logging and monitoring around the affected hosts and services.\n"
        "- Validate authentication controls (MFA, lockout policies, password hygiene).\n"
        "- Contain suspicious endpoints and run malware/EDR scans.\n"
        "- Review firewall and WAF rules related to the observed traffic.\n"
        "- Document findings and escalate according to your incident response plan."
    )


def contains_harmful_request(text: str) -> bool:
    t = text.lower()
    harmful_keywords = [
        "how to hack",
        "exploit",
        "bypass authentication",
        "privilege escalation",
        "zero-day",
        "write malware",
        "develop ransomware",
        "ddos attack",
        "phishing kit",
    ]
    return any(k in t for k in harmful_keywords)


def defensive_only_response() -> Dict[str, str]:
    summary = (
        "The request appears to seek offensive or exploit-focused guidance. "
        "This assistant is restricted to defensive cybersecurity use only."
    )
    attack_type = "Unknown"
    mitre = "No specific technique is provided because the request is offensive in nature."
    defensive_actions = (
        "Focus on hardening systems instead of attacking them:\n"
        "- Apply security patches quickly and maintain an accurate asset inventory.\n"
        "- Enforce strong authentication and least-privilege access.\n"
        "- Monitor logs and alerts from EDR, SIEM, and network sensors.\n"
        "- Conduct regular security training and phishing simulations.\n"
        "- Perform authorized penetration testing under a formal engagement and scope."
    )
    return {
        "summary": summary,
        "attack_type": attack_type,
        "mitre": mitre,
        "defensive_actions": defensive_actions,
    }


def postprocess_model_output(raw_output: str, user_text: str) -> Dict[str, str]:
    combined = raw_output.strip()

    # Try to find explicit sections in the model output.
    def extract_section(header: str) -> str:
        pattern = re.compile(rf"^#+\s*{re.escape(header)}\s*$", re.IGNORECASE | re.MULTILINE)
        m = pattern.search(combined)
        if not m:
            return ""
        start = m.end()
        next_header = re.search(r"^#{1,6}\s", combined[start:], re.MULTILINE)
        end = start + next_header.start() if next_header else len(combined)
        return combined[start:end].strip()

    # Heuristic fallback always uses user_text so output varies by input.
    iocs = extract_iocs(user_text + "\n" + combined)
    heuristic_attack = keyword_fallback_attack_type(user_text)

    summary = extract_section("Summary") or heuristic_summary(user_text, heuristic_attack, iocs)

    model_attack = extract_section("Likely Attack Type")
    attack_label = keyword_fallback_attack_type(model_attack or user_text or combined)

    mitre = extract_section("MITRE Mapping") or map_attack_to_mitre(attack_label)

    defensive = extract_section("Defensive Actions")
    if not defensive:
        defensive = defensive_actions_template(attack_label)

    return {
        "summary": summary,
        "attack_type": attack_label,
        "mitre": mitre,
        "defensive_actions": defensive,
    }


def main() -> None:
    st.set_page_config(page_title="SOC Copilot - cybersec-llm", layout="wide")
    st.title("SOC Copilot (cybersec-llm)")
    st.markdown(
        "Paste logs or security-related text below. The model will attempt to summarize the activity, "
        "guess the likely attack type, suggest a MITRE mapping, extract IOCs, and propose defensive actions."
    )

    user_text = st.text_area(
        "Input logs or security text",
        height=260,
        placeholder="Paste auth logs, web logs, or security alerts here...",
    )

    if st.button("Analyze"):
        if not user_text.strip():
            st.warning("Please provide some text to analyze.")
            return

        if contains_harmful_request(user_text):
            safe = defensive_only_response()
            iocs = extract_iocs(user_text)
            st.markdown("### Summary")
            st.write(safe["summary"])

            st.markdown("### Likely Attack Type (Brute Force / SQLi / XSS / Port Scan / Malware / Suspicious Login / Unknown)")
            st.write(safe["attack_type"])

            st.markdown("### MITRE Mapping (best guess + confidence)")
            st.write(safe["mitre"])

            st.markdown("### IOCs (IPs/domains/urls/hashes/ports extracted with regex)")
            st.json(iocs)

            st.markdown("### Defensive Actions (safe only)")
            st.write(safe["defensive_actions"])
            return

        # Always compute input-dependent heuristic results first (works even when the model is weak).
        base_iocs = extract_iocs(user_text)
        base_attack = keyword_fallback_attack_type(user_text)
        base_mitre = map_attack_to_mitre(base_attack)
        base_summary = heuristic_summary(user_text, base_attack, base_iocs)
        base_defensive = defensive_actions_template(base_attack)

        # Try to load the model; if it fails, we still return useful heuristics.
        model = None
        sp = None
        device = None
        raw_output = ""
        try:
            model, sp, device = load_model_and_tokenizer()
        except Exception:
            model = None

        if model is not None and sp is not None and device is not None:
            prompt = (
                "You are a SOC analyst. Analyze the following logs or security text.\n"
                "Provide a structured response with the following Markdown sections:\n"
                "### Summary\n"
                "### Likely Attack Type (Brute Force / SQLi / XSS / Port Scan / Malware / Suspicious Login / Unknown)\n"
                "### MITRE Mapping (best guess + confidence)\n"
                "### IOCs\n"
                "### Defensive Actions\n\n"
                "Logs:\n"
                f"{user_text}\n\n"
            )

            with st.spinner("Analyzing with cybersec-llm model..."):
                try:
                    raw_output = generate_text(
                        model,
                        sp,
                        device,
                        prompt,
                        max_new_tokens=192,
                        temperature=0.8,
                        top_k=40,
                    )
                except Exception:
                    raw_output = ""

        # Parse model output if present; otherwise use heuristics.
        if raw_output.strip():
            parsed = postprocess_model_output(raw_output, user_text)
            iocs = extract_iocs(user_text + "\n" + raw_output)
        else:
            parsed = {
                "summary": base_summary,
                "attack_type": base_attack,
                "mitre": base_mitre,
                "defensive_actions": base_defensive,
            }
            iocs = base_iocs

        with st.expander("Model raw output (debug)", expanded=False):
            st.code(raw_output or "(no model output)", language="text")

        st.markdown("### Summary")
        st.write(parsed["summary"])

        st.markdown("### Likely Attack Type (Brute Force / SQLi / XSS / Port Scan / Malware / Suspicious Login / Unknown)")
        st.write(parsed["attack_type"])

        st.markdown("### MITRE Mapping (best guess + confidence)")
        st.write(parsed["mitre"])

        st.markdown("### IOCs (IPs/domains/urls/hashes/ports extracted with regex)")
        st.json(iocs)

        st.markdown("### Defensive Actions (safe only)")
        st.write(parsed["defensive_actions"])


if __name__ == "__main__":
    main()

