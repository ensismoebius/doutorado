#!/usr/bin/env python3
import os
import sys
import re

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
exclude_parts = ['build', 'build-coverage', 'build-asan', '_deps', '.git', 'cmake', 'third_party', 'vendor', 'external', 'nfft3-prefix']

extensions = ('.hpp', '.h', '.hh', '.ipp', '.inl', '.cpp', '.cc', '.cxx', '.c')

updated_files = []

def should_skip(path):
    for part in exclude_parts:
        if os.path.sep + part + os.path.sep in path:
            return True
    return False

def has_prologue(text):
    #!/usr/bin/env python3
    import os
    import sys
    import re

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    exclude_parts = ['build', 'build-coverage', 'build-asan', '_deps', '.git', 'cmake', 'third_party', 'vendor', 'external', 'nfft3-prefix']

    extensions = ('.hpp', '.h', '.hh', '.ipp', '.inl', '.cpp', '.cc', '.cxx', '.c')

    updated_files = []

    def should_skip(path):
        for part in exclude_parts:
            if os.path.sep + part + os.path.sep in path:
                return True
        return False

    def has_prologue(text):
        head = '\n'.join(text.splitlines()[:40])
        # Detect existing Doxygen-like prologue or explicit @file/@brief
        if '@file' in head or '@brief' in head or 'PHASE:' in head or 'Canonical C++ API Header' in head:
            return True
        return False

    def make_prologue(relpath, is_header):
        name = os.path.basename(relpath)
        base = os.path.splitext(name)[0]
        # Make a readable description from filename
        desc = re.sub(r'[_\-]+', ' ', base).strip().capitalize()
        if is_header:
            pro = f"/**\n * @file {relpath}\n * @brief {desc}.\n *\n * \n *\n * **Contract:**\n * - Public APIs should document behavior, inputs, outputs, and exceptions.\n * - Prefer RAII for resource lifecycle when applicable.\n */\n\n"
        else:
            pro = f"/**\n * @file {relpath}\n * @brief Implementation for {desc}.\n *\n \n */\n\n"
        return pro

    def insert_section_header_if_missing(text):
        # Skip if there's already a section marker
        if '// -----------------------------------------------------------------' in text:
            return text, False

        # Find first class/struct occurrence (outside the prologue area)
        m = re.search(r"\n(?:template\s*<[^>]*>\s*)?(class|struct)\s+\w", text, flags=re.M)
        if not m:
            return text, False

        insert_at = m.start()  # position at the start of the match
        section = '\n// -----------------------------------------------------------------\n// Public API\n// -----------------------------------------------------------------\n'
        new_text = text[:insert_at] + section + text[insert_at:]
        return new_text, True

    for root, dirs, files in os.walk(repo_root):
        # Skip excluded directories early
        if should_skip(root + os.path.sep):
            continue
        for fname in files:
            if not fname.endswith(extensions):
                continue
            path = os.path.join(root, fname)
            # Skip files in hidden dirs
            if '/.' in path:
                continue
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception:
                continue
            if len(text.strip()) == 0:
                continue
            rel = os.path.relpath(path, repo_root).replace('\\','/')
            changed = False
            # Add prologue if missing
            if not has_prologue(text):
                is_header = fname.endswith(('.hpp', '.h', '.hh', '.ipp', '.inl'))
                pro = make_prologue(rel, is_header)
                text = pro + text
                changed = True

            # For headers, insert a standard section header before first class/struct
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

    print('\nSummary:')
    print(f'Files updated: {len(updated_files)}')
    for f in updated_files[:500]:
        print(f)

    sys.exit(0)
