#!/usr/bin/env python3
"""Reject private paths, excluded research, credentials and unintended cross-repository imports."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATTERNS = [
    r'/home/', r'/data1', r'\bjokim\b(?!@snu\.ac\.kr)', r'\bARIS\b',
    r'(?i)\bicml\b|locality|offsethome|overshoot|F50[12]',
    r'(?i)gpt-|codex|claude|Anthropic|ACS_ROBI',
    r'(?i)(?:gh[pousr]_|github_pat_|sk-[A-Za-z0-9])[A-Za-z0-9_\-]{16,}',
    r'AKIA[0-9A-Z]{16}', r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    r'(?i)(?:token|password|secret)\s*=\s*[\"\'][A-Za-z0-9+/=_\-]{20,}[\"\']',
]
# This checker names its own patterns; ignore rules exclude local operator tool directories.
CLASSIFIED = {'scripts/bench/check_self_contained.py', '.gitignore'}


def main():
    files = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z']).decode().split('\0')
    findings = []
    classified = 0
    emails = re.compile(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}')
    for rel in filter(None, files):
        path = ROOT / rel
        if path.is_dir() or not path.is_file():
            continue
        data = path.read_bytes()
        if b'\0' in data:
            continue
        for number, line in enumerate(data.decode(errors='replace').splitlines(), 1):
            bad = any(re.search(p, line) for p in PATTERNS)
            bad |= any(email != 'jokim@snu.ac.kr' for email in emails.findall(line))
            if bad:
                if rel in CLASSIFIED:
                    classified += 1
                else:
                    findings.append(f'{rel}:{number}: {line[:140]}')
    for finding in findings:
        print(finding)
    print(f'{len(findings)} unclassified; {classified} classified checker/ignore-rule lines')
    return bool(findings)

if __name__ == '__main__':
    raise SystemExit(main())
