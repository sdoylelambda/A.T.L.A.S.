"""
modules/security_audit.py

A.T.L.A.S. Security Audit Module

Checks the health of Atlas's own remote-access security posture —
SSH config, listening ports, Tailscale status, firewall, failed logins,
and pending system updates. Compares against a saved baseline to detect
configuration drift over time.

This is NOT a penetration testing tool. It answers one question:
"Is my remote access configuration still what I expect it to be?"

Voice commands (via security_handler.py):
    "verify connection security"
    "run security audit"
    "check security"

Author: Sean Doyle / A.T.L.A.S. Project
"""

import json
import re
import subprocess
from pathlib import Path
from datetime import datetime

BASELINE_PATH = Path.home() / ".atlas" / "security_baseline.json"

# Ports Atlas expects to see open — adjust to match your setup
EXPECTED_PORTS = {
    22:    "ssh",
    8000:  "atlas api",
    41641: "tailscale",
    11434: "ollama"
}

# SSH settings we want to see, and what "good" looks like
SSH_RECOMMENDED = {
    "passwordauthentication": "no",   # we WANT password auth too for 2FA — see note below
    "permitrootlogin":        "no",
    "permitemptypasswords":   "no",
    "x11forwarding":          "no",
}


def _run(cmd: list, timeout: int = 10) -> tuple[bool, str]:
    """Run a command, return (success, output). Never raises."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0 and not result.stdout:
            return False, result.stderr.strip()
        return True, result.stdout.strip()
    except FileNotFoundError:
        return False, f"{cmd[0]} not installed"
    except subprocess.TimeoutExpired:
        return False, "timed out"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_ssh_config() -> dict:
    """Parse sshd_config directly — avoids needing root for host keys."""
    config_paths = [Path("/etc/ssh/sshd_config")]

    # Ubuntu/Pop!_OS split configs into sshd_config.d/
    drop_in_dir = Path("/etc/ssh/sshd_config.d")
    if drop_in_dir.exists():
        config_paths.extend(sorted(drop_in_dir.glob("*.conf")))

    settings = {}
    # OpenSSH defaults — used when a setting isn't explicitly set
    defaults = {
        "passwordauthentication": "yes",
        "permitrootlogin":        "prohibit-password",
        "permitemptypasswords":   "no",
        "x11forwarding":          "no",
    }
    settings.update(defaults)

    found_any = False
    for path in config_paths:
        try:
            text = path.read_text()
            found_any = True
        except (PermissionError, FileNotFoundError):
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                settings[parts[0].lower()] = parts[1].lower()

    if not found_any:
        return {"status": "unknown", "detail": "Could not read sshd_config (permission denied)"}

    findings = []
    warnings = []

    # permitrootlogin: "no" or "prohibit-password" are both fine
    root_login = settings.get("permitrootlogin")
    if root_login in ("no", "prohibit-password"):
        findings.append(f"permitrootlogin = {root_login} (good)")
    else:
        warnings.append(f"permitrootlogin = {root_login} (recommend: no)")

    empty_pw = settings.get("permitemptypasswords")
    if empty_pw == "no":
        findings.append("permitemptypasswords = no (good)")
    else:
        warnings.append(f"permitemptypasswords = {empty_pw} (recommend: no)")

    x11 = settings.get("x11forwarding")
    if x11 == "no":
        findings.append("x11forwarding = no (good)")
    else:
        warnings.append(f"x11forwarding = {x11} (recommend: no)")

    pw_auth = settings.get("passwordauthentication")
    note = None
    if pw_auth == "yes":
        note = "Password authentication enabled — intentional for two-factor SSH+password setup."
        findings.append("passwordauthentication = yes (intentional — see note)")

    return {
        "status":   "warning" if warnings else "good",
        "findings": findings,
        "warnings": warnings,
        "note":     note,
        "raw":      settings,
    }

KNOWN_SAFE_PROCESSES = {
    "avahi-daemon":     "mDNS / device discovery — normal on Linux desktops",
    "systemd-resolve":  "DNS resolver stub — normal",
    "systemd-resolved": "DNS resolver stub — normal",
    "cupsd":            "Printing service — normal if you use a printer",
    "cups-browsed":     "CUPS network printer discovery — normal",
    "ollama":           "Atlas's local LLM backend — expected",
    "chrome":           "Chrome browser — ephemeral ports, normal",
    "code":             "VS Code — ephemeral ports, normal",
    "dart":             "Flutter/Dart dev tooling — normal during development",
    "adb":              "Android Debug Bridge — normal during development",
    "kdeconnectd":      "KDE Connect / phone integration",
    "NetworkManager":   "Network management — normal",
    "dnsmasq":          "DHCP/DNS for VM or container networking",
    "dhclient":         "DHCP client — normal",
    "chronyd":          "Time sync — normal",
    "firefox-bin": "Firefox — likely WebRTC ephemeral ports from a video/voice call tab, normal",
    "firefox":     "Firefox — likely WebRTC ephemeral ports from a video/voice call tab, normal",
}


def _dedupe(seq: list) -> list:
    seen, out = set(), []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def check_listening_ports() -> dict:
    ok, output = _run(["sudo", "-n", "ss", "-tulnp"])
    used_sudo = ok
    if not ok:
        ok, output = _run(["ss", "-tuln"])
        if not ok:
            return {"status": "unknown", "detail": f"Could not list ports: {output}"}

    findings, warnings, info = [], [], []
    open_ports = set()
    exposed_ports = []   # ← only genuinely risky ports go here

    def _is_loopback(addr: str) -> bool:
        a = addr.strip("[]")
        return (
                a.startswith("127.") or
                a in ("::1", "localhost") or
                a.startswith("::ffff:127.")
        )

    def _is_tailscale_addr(addr: str) -> bool:
        a = addr.strip("[]")
        return a.startswith("100.") or a.startswith("fd7a:115c:a1e0")

    for line in output.splitlines()[1:]:
        cols = line.split()
        if len(cols) < 5:
            continue
        local = cols[4]
        if ":" not in local:
            continue
        addr, _, port_str = local.rpartition(":")
        if not port_str.isdigit():
            continue
        port = int(port_str)
        open_ports.add(port)

        proc_name = None
        if used_sudo and len(cols) >= 6:
            m = re.search(r'\("([^"]+)"', cols[-1])
            if m:
                proc_name = m.group(1)

        if port in EXPECTED_PORTS:
            findings.append(f"Port {port} ({EXPECTED_PORTS[port]}) — expected")
            continue

        label = f"Port {port}" + (f" ({proc_name})" if proc_name else "")

        if proc_name and proc_name in KNOWN_SAFE_PROCESSES:
            info.append(f"{label} — {KNOWN_SAFE_PROCESSES[proc_name]}")
        elif _is_loopback(addr):
            info.append(f"{label} — loopback only, not network exposed")
        elif _is_tailscale_addr(addr):
            info.append(f"{label} — bound to Tailscale interface only")
        elif addr.startswith("[fe80") or addr.startswith("fe80"):
            info.append(f"{label} — link-local IPv6, not routable beyond this network segment")
        elif addr in ("*", "0.0.0.0", "[::]", "::"):
            warnings.append(f"{label} — open on ALL interfaces (worth checking)")
            exposed_ports.append(port)
        else:
            warnings.append(f"{label} — open on {addr} (LAN-reachable, worth checking)")
            exposed_ports.append(port)

    missing = sorted(p for p in EXPECTED_PORTS if p not in open_ports)
    for port in missing:
        warnings.append(f"Expected port not listening: {port} ({EXPECTED_PORTS[port]})")

    return {
        "status":        "warning" if warnings else "good",
        "findings":      _dedupe(findings),
        "warnings":      _dedupe(warnings),
        "info":          _dedupe(info),
        "open_ports":    sorted(open_ports),
        "exposed_ports": sorted(set(exposed_ports)),
        "process_lookup_available": used_sudo,
    }

def check_tailscale() -> dict:
    """Check Tailscale connection status."""
    ok, output = _run(["tailscale", "status", "--json"])
    if not ok:
        return {"status": "unknown", "detail": f"Could not check Tailscale: {output}"}

    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return {"status": "unknown", "detail": "Could not parse Tailscale status"}

    self_info  = data.get("Self", {})
    warnings   = []
    findings   = []

    if self_info.get("Online"):
        findings.append("Tailscale connected")
    else:
        warnings.append("Tailscale is not connected")

    # check for risky features
    if data.get("CurrentTailnet", {}).get("MagicDNSEnabled"):
        findings.append("MagicDNS enabled")

    exit_node = data.get("ExitNodeStatus")
    if exit_node:
        warnings.append("Exit node is active — traffic may be routed through another device")

    return {
        "status":   "warning" if warnings else "good",
        "findings": findings,
        "warnings": warnings,
    }


def check_firewall() -> dict:
    """Check ufw status. Requires sudo — degrades gracefully if unavailable."""
    ok, output = _run(["sudo", "-n", "ufw", "status", "verbose"])
    if not ok:
        return {
            "status": "unknown",
            "detail": "Could not check firewall (requires sudo without password prompt). "
                      "Run 'sudo visudo' to allow passwordless ufw status if you want this check.",
        }

    if "inactive" in output.lower():
        return {"status": "warning", "warnings": ["Firewall (ufw) is inactive"]}

    findings = ["Firewall (ufw) is active"]
    if "deny (incoming)" in output.lower():
        findings.append("Default deny incoming — good")

    return {"status": "good", "findings": findings, "raw": output}


def check_pending_updates() -> dict:
    """Check for available system updates."""
    ok, output = _run(["apt", "list", "--upgradable"], timeout=15)
    if not ok:
        return {"status": "unknown", "detail": "Could not check for updates"}

    lines = [l for l in output.splitlines() if "/" in l and "Listing" not in l]
    count = len(lines)

    if count == 0:
        return {"status": "good", "findings": ["System is up to date"]}
    return {
        "status":   "warning",
        "warnings": [f"{count} package(s) have updates available"],
        "count":    count,
    }


def check_failed_logins() -> dict:
    """Check recent failed SSH login attempts. Requires journal access."""
    ok, output = _run(
        ["journalctl", "-u", "ssh", "--since", "24 hours ago", "--no-pager"],
        timeout=10,
    )
    if not ok:
        return {"status": "unknown", "detail": "Could not check login history"}

    failed = [l for l in output.splitlines() if "Failed password" in l or "Invalid user" in l]

    if not failed:
        return {"status": "good", "findings": ["No failed login attempts in last 24 hours"]}
    return {
        "status":   "warning",
        "warnings": [f"{len(failed)} failed login attempt(s) in last 24 hours"],
        "count":    len(failed),
    }


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def _build_snapshot(results: dict) -> dict:
    """Extract the comparable facts from a full audit for baseline storage."""
    return {
        "exposed_ports": results["ports"].get("exposed_ports", []),
        "ssh_settings": results["ssh"].get("raw", {}),
        "tailscale_online": "Tailscale connected" in results["tailscale"].get("findings", []),
        "timestamp": datetime.now().isoformat(),
    }


def _load_baseline() -> dict | None:
    if BASELINE_PATH.exists():
        try:
            return json.loads(BASELINE_PATH.read_text())
        except json.JSONDecodeError:
            return None
    return None


def _save_baseline(snapshot: dict):
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(snapshot, indent=2))


def _diff_baseline(old: dict, new: dict) -> list[str]:
    changes = []

    old_ports = set(old.get("exposed_ports", []))
    new_ports = set(new.get("exposed_ports", []))
    added   = new_ports - old_ports
    removed = old_ports - new_ports
    if added:
        changes.append(f"New exposed port(s) — worth investigating: {sorted(added)}")
    if removed:
        changes.append(f"Previously exposed port(s) no longer open: {sorted(removed)}")

    old_ssh = old.get("ssh_settings", {})
    new_ssh = new.get("ssh_settings", {})
    for key in SSH_RECOMMENDED:
        if key in old_ssh and key in new_ssh and old_ssh[key] != new_ssh[key]:
            changes.append(f"SSH setting changed: {key} went from {old_ssh[key]} to {new_ssh[key]}")

    if old.get("tailscale_online") and not new.get("tailscale_online"):
        changes.append("Tailscale went offline since last check")

    return changes


# ---------------------------------------------------------------------------
# Full audit
# ---------------------------------------------------------------------------

def run_audit(save_baseline: bool = True) -> dict:
    """
    Run the full security audit. Returns structured results plus
    a voice-friendly summary and a 0-100 score.
    """
    results = {
        "ssh":       check_ssh_config(),
        "ports":     check_listening_ports(),
        "tailscale": check_tailscale(),
        "firewall":  check_firewall(),
        "updates":   check_pending_updates(),
        "logins":    check_failed_logins(),
    }

    # scoring — each check starts at its share of 100, deduct for warnings
    weights = {
        "ssh": 25, "ports": 25, "tailscale": 20,
        "firewall": 15, "updates": 10, "logins": 5,
    }
    score = 0
    all_warnings = []
    unknown_checks = []

    for name, weight in weights.items():
        status = results[name]["status"]
        if status == "good":
            score += weight
        elif status == "warning":
            score += weight * 0.5
            all_warnings.extend(results[name].get("warnings", []))
        else:  # unknown
            score += weight * 0.75  # don't punish hard for unknown
            unknown_checks.append(name)

    score = round(score)

    # baseline diff
    snapshot = _build_snapshot(results)
    baseline = _load_baseline()
    drift = []
    if baseline:
        drift = _diff_baseline(baseline, snapshot)
    if save_baseline:
        _save_baseline(snapshot)

    return {
        "score":           score,
        "results":         results,
        "warnings":        all_warnings,
        "drift":            drift,
        "unknown_checks":  unknown_checks,
        "is_first_run":    baseline is None,
        "timestamp":       datetime.now().isoformat(),
    }


def format_voice_summary(audit: dict) -> str:
    """Produce a short spoken summary of the audit results."""
    score = audit["score"]
    parts = [f"Security audit complete. Score: {score} out of 100."]

    if audit["is_first_run"]:
        parts.append("This is the first audit — baseline saved for future comparison.")
    elif audit["drift"]:
        parts.append(f"{len(audit['drift'])} change(s) detected since last check.")
        for change in audit["drift"][:3]:
            parts.append(change + ".")
    else:
        parts.append("No changes detected since the last check.")

    if audit["warnings"]:
        parts.append(f"{len(audit['warnings'])} item(s) need attention.")
        for w in audit["warnings"][:2]:
            parts.append(w + ".")
    else:
        parts.append("No active warnings.")

    if audit["unknown_checks"]:
        parts.append(
            f"Could not fully check: {', '.join(audit['unknown_checks'])}."
        )

    return " ".join(parts)


def format_full_report(audit: dict) -> str:
    """Produce a detailed text report — for display or logging."""
    lines = [
        "=" * 50,
        f"A.T.L.A.S. Security Audit — Score: {audit['score']}/100",
        f"Timestamp: {audit['timestamp']}",
        "=" * 50,
    ]

    for name, result in audit["results"].items():
        status_icon = {"good": "✓", "warning": "⚠", "unknown": "?"}.get(
            result["status"], "?")
        lines.append(f"\n[{status_icon}] {name.upper()}")
        for f in result.get("findings", []):
            lines.append(f"    ✓ {f}")
        for i in result.get("info", []):
            lines.append(f"    · {i}")
        for w in result.get("warnings", []):
            lines.append(f"    ⚠ {w}")
        if result.get("note"):
            lines.append(f"    ℹ {result['note']}")
        if result.get("detail"):
            lines.append(f"    ? {result['detail']}")

    if audit["drift"]:
        lines.append(f"\n[CHANGES SINCE LAST CHECK]")
        for d in audit["drift"]:
            lines.append(f"    • {d}")

    lines.append("\n" + "=" * 50)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    audit = run_audit()
    print(format_full_report(audit))
    print("\nVoice summary:")
    print(format_voice_summary(audit))