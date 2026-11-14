import unittest
from automated_repair import classify_error, extract_code, repair

TESTS = 'from candidate import square\nassert square(3) == 9\nassert square(-4) == 16\n'

class RepairTests(unittest.TestCase):
    def test_baseline_pass(self):
        result = repair('def square(x): return x*x\n', TESTS, lambda *_: self.fail('proposer should not run'))
        self.assertTrue(result['success'])
        self.assertEqual(result['repairs_used'], 0)

    def test_repair_success(self):
        result = repair('def square(x): return x+x\n', TESTS, lambda *_: 'def square(x): return x*x\n')
        self.assertTrue(result['success'])
        self.assertEqual(result['repairs_used'], 1)
        self.assertEqual(result['attempts'][0]['category'], 'correctness')

    def test_exhausted(self):
        result = repair('def square(x): return 0\n', TESTS, lambda src, *_: src, max_repairs=2)
        self.assertFalse(result['success'])
        self.assertEqual(len(result['attempts']), 3)

    def test_syntax_failure(self):
        result = repair('def square(:\n', TESTS, lambda *_: 'def square(x): return x*x\n')
        self.assertEqual(result['attempts'][0]['category'], 'syntax')
        self.assertTrue(result['success'])

    def test_error_classification(self):
        self.assertEqual(classify_error('ValueError: incompatible shapes'), 'shape')
        self.assertEqual(classify_error('ConcretizationTypeError'), 'jax_tracing')

    def test_code_fences(self):
        self.assertEqual(extract_code('```python\nx = 1\n```').strip(), 'x = 1')

    def test_empty_proposal(self):
        with self.assertRaises(ValueError):
            repair('def square(x): return 0\n', TESTS, lambda *_: '')

if __name__ == '__main__':
    unittest.main()
