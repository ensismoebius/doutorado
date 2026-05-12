#!/usr/bin/env python3
"""apply_header_style.py — Add Doxygen file prologues and section markers.

Walks the entire repo tree (from repo root) and, for every C/C++ source or
header that lacks a @file/@brief prologue, prepends one.  For headers it also
inserts a '// Public API' section divider before the first class/struct.

Skips: build/, out/, _deps/, .git/, cmake/, third_party/, vendor/, external/,
       nfft3-prefix/, and any path containing a hidden directory.

Usage:
    python scripts/dev/apply_header_style.py

Run from any directory; the script resolves the repo root via __file__.
"""
import os
import sys
import re

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
exclude_parts = ['build', 'build-coverage', 'build-asan', '_deps', '.git', 'cmake',
                 'third_party', 'vendor', 'external', 'nfft3-prefix', 'out']

extensions = ('.hpp', '.h', '.hh', '.ipp', '.inl', '.cpp', '.cc', '.cxx', '.c')

updated_files = []


def should_skip(path):
    for part in exclude_parts:
        if os.path.sep + part + os.path.sep in path:
            return True
    return False


def has_prologue(text):
    head = '\n'.join(text.splitlines()[:40])
    return '@file' in head or '@brief' in head or 'PHASE:' in head or 'Canonical C++ API Header' in head


def make_prologue(relpath, is_header):
    name = os.path.basename(relpath)
    base = os.path.splitext(name)[0]
    desc = re.sub(r'[_\-]+', ' ', base).strip().capitalize()
    if is_header:
        return (f"/**\n * @file {relpath}\n * @brief {desc}.\n *\n * \n *\n"
                f" * **Contract:**\n"
                f" * - Public APIs should document behavior, inputs, outputs, and exceptions.\n"
                f" * - Prefer RAII for resource lifecycle when applicable.\n */\n\n")
    return f"/**\n * @file {relpath}\n * @brief Implementation for {desc}.\n *\n \n */\n\n"


def insert_section_header_if_missing(text):
    if '// -----------------------------------------------------------------' in text:
        return text, False
    m = re.search(r"\n(?:template\s*<[^>]*>\s*)?(class|struct)\s+\w", text, flags=re.M)
    if not m:
        return text, False
    insert_at = m.start()
    section = '\n// -----------------------------------------------------------------\n// Public API\n// -----------------------------------------------------------------\n'
    return text[:insert_at] + section + text[insert_at:], True


for root, dirs, files in os.walk(repo_root):
    if should_skip(root + os.path.sep):
        continue
    for fname in files:
        if not fname.endswith(extensions):
            continue
        path = os.path.join(root, fname)
        if '/.' in path:
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception:
            continue
        if not text.strip():
            continue
        rel = os.path.relpath(path, repo_root).replace('\\', '/')
        changed = False
        if not has_prologue(text):
            is_header = fname.endswith(('.hpp', '.h', '.hh', '.ipp', '.inl'))
            text = make_prologue(rel, is_header) + text
            changed = True
        if fname.endswith(('.hpp', '.h', '.hh', '.ipp', '.inl')):
            new_text, inserted = insert_section_header_if_missing(text)
            if inserted:
                text = new_text
                changed = True
        if changed:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(text)
                updated_files.append(rel)
                print(f"Updated: {rel}")
            except Exception as e:
                print(f"ERROR writing {rel}: {e}", file=sys.stderr)

print(f'\nSummary: {len(updated_files)} files updated')
for f in updated_files[:500]:
    print(f)

sys.exit(0)
