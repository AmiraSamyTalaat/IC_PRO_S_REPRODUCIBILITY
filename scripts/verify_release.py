#!/usr/bin/env python3
"""Immutable release file hashes. Not a scientific rerun."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=root/'RELEASE_SHA256.json'
if not p.is_file():raise SystemExit('Release checksum manifest missing')
items=json.loads(p.read_text(encoding='utf-8'))['files']
fail=[]
for rel,h in items.items():
 q=root/rel
 if not q.is_file():fail.append('MISSING '+rel)
 elif hashlib.sha256(q.read_bytes()).hexdigest()!=h:fail.append('HASH '+rel)
if fail:raise SystemExit('\n'.join(fail))
print('PASS:',len(items),'files verified against release manifest; this is byte integrity, not full scientific reproduction')
