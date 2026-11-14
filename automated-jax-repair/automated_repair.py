"""Experimental test-driven repair loop. Run untrusted code only in an isolated VM/container."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib import request

def classify_error(output):
    s = output.lower()
    if 'syntaxerror' in s or 'indentationerror' in s: return 'syntax'
    if 'modulenotfounderror' in s or 'importerror' in s: return 'dependency'
    if any(x in s for x in ('shape', 'broadcast', 'dimension', 'incompatible')): return 'shape'
    if 'tracer' in s or 'concretizationtypeerror' in s: return 'jax_tracing'
    if 'attributeerror' in s or 'nameerror' in s: return 'api_or_name'
    if 'assertionerror' in s: return 'correctness'
    if 'timed out' in s: return 'timeout'
    return 'other'

def run_tests(candidate, test_source, timeout=20):
    with tempfile.TemporaryDirectory(prefix='jax_repair_') as tmp:
        root = Path(tmp)
        (root / 'candidate.py').write_text(candidate, encoding='utf-8')
        (root / 'test_candidate.py').write_text(test_source, encoding='utf-8')
        try:
            p = subprocess.run([sys.executable, '-B', str(root / 'test_candidate.py')],
                               cwd=root, capture_output=True, text=True, timeout=timeout,
                               env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            output = (p.stdout + '\n' + p.stderr)[-12000:]
            return {'passed': p.returncode == 0, 'returncode': p.returncode,
                    'output': output, 'category': None if p.returncode == 0 else classify_error(output)}
        except subprocess.TimeoutExpired:
            return {'passed': False, 'returncode': None,
                    'output': f'Test timed out after {timeout}s', 'category': 'timeout'}

def extract_code(text):
    if '```' in text:
        parts = text.split('```')
        if len(parts) >= 3:
            block = parts[1]
            return block.split('\n', 1)[1] if block.startswith(('python\n', 'py\n')) else block
    return text

def openai_repair(source, feedback, test_source, model):
    key = os.environ.get('OPENAI_API_KEY')
    if not key: raise RuntimeError('OPENAI_API_KEY is required for live repairs')
    payload = {'model': model, 'temperature': 0, 'messages': [
        {'role': 'system', 'content': 'Repair the Python/JAX candidate. Return only complete Python source, no explanation. Preserve public function names and signatures.'},
        {'role': 'user', 'content': f'Candidate:\n```python\n{source}\n```\n\nTest source:\n```python\n{test_source}\n```\n\nFailure:\n{feedback}'}
    ]}
    req = request.Request('https://api.openai.com/v1/chat/completions',
        data=json.dumps(payload).encode(),
        headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
        method='POST')
    with request.urlopen(req, timeout=90) as response:
        result = json.load(response)
    return extract_code(result['choices'][0]['message']['content']).strip() + '\n'

def repair(source, tests, proposer, max_repairs=3, timeout=20):
    if max_repairs < 0: raise ValueError('max_repairs must be nonnegative')
    current, attempts = source, []
    for n in range(max_repairs + 1):
        result = run_tests(current, tests, timeout)
        attempts.append({'attempt': n, **result})
        if result['passed']:
            return {'success': True, 'repairs_used': n, 'attempts': attempts, 'final_source': current}
        if n < max_repairs:
            proposed = proposer(current, result['output'], tests)
            if not proposed.strip(): raise ValueError('Repair proposer returned empty source')
            current = proposed
    return {'success': False, 'repairs_used': max_repairs, 'attempts': attempts, 'final_source': current}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--tests', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', default='gpt-4o-mini')
    parser.add_argument('--max-repairs', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=20)
    args = parser.parse_args()
    result = repair(args.candidate.read_text(encoding='utf-8'),
        args.tests.read_text(encoding='utf-8'),
        lambda src, err, tests: openai_repair(src, err, tests, args.model),
        max_repairs=args.max_repairs, timeout=args.timeout)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'repaired.py').write_text(result.pop('final_source'), encoding='utf-8')
    (args.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not result['success']: sys.exit(1)

if __name__ == '__main__':
    main()
