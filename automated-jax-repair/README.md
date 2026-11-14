# Automated JAX Repair Prototype

Experimental test-driven repair loop extending the manually repaired TorchLeet/JAX examples.

## What it does

1. Runs generated Python/JAX code against a supplied assertion-based test script.
2. Captures stderr/stdout and classifies the failure.
3. Requests a full corrected candidate from an LLM.
4. Repeats up to three repair attempts.
5. Saves the final candidate and a JSON attempt log.

## Unit tests

From this directory:

```bash
python -m unittest discover -s . -p 'test_*.py' -v
```

Seven local orchestration unit tests passed before upload. These use pure-Python mock repair proposals, **not** a live LLM or TorchLeet benchmark.

## Live usage

Provide `candidate.py` and `tests.py` (which imports `candidate` and asserts correctness). Set `OPENAI_API_KEY` in the environment (never commit secrets).

```bash
python automated_repair.py --candidate candidate.py --tests tests.py --output results/example --max-repairs 3
```

The default live provider is OpenAI Chat Completions. For Qwen experiments, replace `openai_repair` with a compatible local-model proposer.

**Security:** Generated code must only run inside a disposable, isolated VM/container with no secrets, no network access and resource limits. The subprocess timeout is **not** a sandbox; the test runner does not enforce memory limits or terminate subprocess descendants. Separate the networked LLM caller from the sandbox.

## Current evaluation status

- Repair orchestration unit tests: 7/7 passed locally.
- Real Qwen repairs: not evaluated.
- BPE, Embeddings, GQA, Flash Attention numerical equivalence: not evaluated by this prototype.
- Live API call: not tested.

Next: recover original broken Qwen outputs and manually fixed references; write numerical equivalence tests; measure baseline pass rate, repaired pass rate, and attempts per problem. Do not report benchmark success until measured.
