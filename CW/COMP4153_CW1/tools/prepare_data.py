"""Download original public data once; validate existing files before reuse."""
from pathlib import Path
import csv
import hashlib
import io
import json
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
SOURCES = {
    'banking77/train.csv': 'https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv',
    'banking77/test.csv': 'https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv',
    'banking77/categories.json': 'https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/categories.json',
    'clinc150/data_full.json': 'https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json',
}


def validate(name, raw):
    if name.endswith('.csv'):
        rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
        assert len(rows) == (10003 if name.endswith('train.csv') else 3080)
        assert all(r.get('text') and r.get('category') for r in rows)
        assert len({r['category'] for r in rows}) == 77
        return {'rows': len(rows), 'classes': 77}
    obj = json.loads(raw)
    if name.endswith('categories.json'):
        assert len(obj) == len(set(obj)) == 77
        return {'classes': 77}
    expected = {'train': 15000, 'val': 3000, 'test': 4500,
                'oos_train': 100, 'oos_val': 100, 'oos_test': 1000}
    for split, size in expected.items():
        assert len(obj[split]) == size, (split, len(obj[split]))
        assert all(isinstance(x, list) and len(x) == 2 and x[0] for x in obj[split])
    labels = {y for _, y in obj['train']}
    assert len(labels) == 150
    assert all({y for _, y in obj[s]} == labels for s in ['val', 'test'])
    return {'splits': expected, 'in_domain_classes': 150, 'oos_kept_separate': True}


def main():
    manifest = {}
    for name, url in SOURCES.items():
        path = DATA / name
        reused = path.exists()
        raw = path.read_bytes() if reused else urllib.request.urlopen(url, timeout=90).read()
        info = validate(name, raw)
        if not reused:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        manifest[name] = dict(url=url, sha256=hashlib.sha256(raw).hexdigest(),
                              reused=reused, bytes=len(raw), **info)
        print(name, 'REUSED' if reused else 'DOWNLOADED', info, flush=True)
    manifest['checked_utc'] = datetime.now(timezone.utc).isoformat()
    (DATA / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
