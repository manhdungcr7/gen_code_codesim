from typing import List
from .Dataset import Dataset
from evaluations.func_evaluate import evaluate_functional_correctness, evaluate_io

class BigCodeBenchDataset(Dataset):
    def __init__(self, path: str = "data/BigCodeBench/BigCodeBench-Hard.jsonl"):
        super().__init__(path)
        self.id_key = "task_id"

    def evaluate(self, item: dict, cur_imp: str, language: str):
        # BCB uses unittest.TestCase, not HumanEval-style check(entry_point).
        # Append a proper unittest runner so the sandbox exits non-zero on failure.
        test_with_runner = (
            item["test"]
            + "\nimport unittest"
            + "\n_suite = unittest.TestLoader().loadTestsFromTestCase(TestCases)"
            + "\n_result = unittest.TextTestRunner(verbosity=0).run(_suite)"
            + "\nif not _result.wasSuccessful(): exit(1)"
        )
        result = evaluate_functional_correctness(
            test=test_with_runner,
            entry_point="",  # Don't append check() — unittest runner handles it
            completion=cur_imp,
        )
        return result == "passed"

    def evaluate_sample_io(self, item: dict, cur_imp: str, language: str):
        # BCB Hard has no sample_io. Run first 2 unittest methods as feedback
        # so CodeSIM's debug loop can trigger when the code is wrong.
        test_with_runner = (
            item["test"]
            + "\nimport unittest"
            + "\n_loader = unittest.TestLoader()"
            + "\n_names = _loader.getTestCaseNames(TestCases)[:2]"
            + "\n_suite = unittest.TestSuite(TestCases(n) for n in _names)"
            + "\n_result = unittest.TextTestRunner(verbosity=0).run(_suite)"
            + "\nif not _result.wasSuccessful(): exit(1)"
        )
        return evaluate_io(sample_io=[test_with_runner], completion=cur_imp)
    
    def evaluate_additional_io(self, id: int, io: List[str], cur_imp: str, language: str):
        if len(io) == 0:
            return True, ""
        return evaluate_io(sample_io=io, completion=cur_imp)

    @staticmethod
    def get_prompt(item):
        if "prompt" in item:
            return f"{item['prompt'].strip()}"
        else:
            raise Exception("No prompt in item")