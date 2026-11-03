"""Infrastructure diagnostics only; no coursework experiment results."""
from pathlib import Path
import importlib
import importlib.metadata as metadata
import json
import subprocess
import sys
import traceback
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'data' / '.cache' / 'matplotlib'))
PACKAGES = {'torch': 'torch', 'torchvision': 'torchvision', 'setfit': 'setfit',
            'sentence-transformers': 'sentence_transformers', 'transformers': 'transformers',
            'datasets': 'datasets', 'accelerate': 'accelerate', 'evaluate': 'evaluate',
            'scikit-learn': 'sklearn', 'pandas': 'pandas', 'numpy': 'numpy',
            'matplotlib': 'matplotlib', 'seaborn': 'seaborn', 'ipykernel': 'ipykernel',
            'jupyterlab': 'jupyterlab'}
report = {'python': sys.version, 'executable': sys.executable, 'packages': {}}
for package, module in PACKAGES.items():
    try:
        importlib.import_module(module)
        report['packages'][package] = {'version': metadata.version(package), 'import': 'PASS'}
    except Exception:
        report['packages'][package] = {'import': 'FAIL', 'error': traceback.format_exc()}
check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
report['pip_check'] = {'returncode': check.returncode, 'output': check.stdout + check.stderr}
try:
    import torch
    x = torch.randn(64, 64, device='cuda', requires_grad=True)
    (x @ x).sum().backward()
    torch.cuda.synchronize()
    report['gpu'] = {'name': torch.cuda.get_device_name(0), 'runtime': torch.version.cuda,
                     'capability': torch.cuda.get_device_capability(0),
                     'architectures': torch.cuda.get_arch_list(), 'forward_backward': 'PASS',
                     'vram_bytes': torch.cuda.get_device_properties(0).total_memory}
except Exception:
    report['gpu'] = {'error': traceback.format_exc()}
out = ROOT / 'results' / 'environment_check.json'
out.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2), flush=True)
if check.returncode or any(x['import'] != 'PASS' for x in report['packages'].values()) or 'error' in report['gpu']:
    raise SystemExit(1)
