import base64
import zlib
import pickle
import json
from typing import List
from datasets.Dataset import Dataset
from evaluations.func_evaluate import evaluate_functional_correctness, evaluate_io

def decode_lbpp_field(val):
    if not isinstance(val, str):
        return val
    try:
        unpickled = pickle.loads(zlib.decompress(base64.b64decode(val)))
        try:
            return json.loads(unpickled)
        except Exception:
            return unpickled
    except Exception:
        return val

class LBPPDataset(Dataset):
    def __init__(self, path: str = "data/LBPP/lbpp.jsonl"):
        super().__init__(path)
        self.id_key = "task_id"

    def _decode_item(self, item: dict):
        setup = decode_lbpp_field(item.get("test_setup", ""))
        test_list = decode_lbpp_field(item.get("test_list", []))
        
        if isinstance(setup, str):
            lines = [l for l in setup.split('\n') if not (l.startswith("from code ") or l.startswith("from solution "))]
            setup = "\n".join(lines)
            
        return setup, test_list
            
    def evaluate(self, item: dict, cur_imp: str, language: str):
        setup, test_list = self._decode_item(item)
        tests = "\n".join(test_list) if isinstance(test_list, list) else str(test_list)
        test_block = setup + "\n" + tests
        
        result = evaluate_functional_correctness(
            test=test_block,
            entry_point="", 
            completion=cur_imp,
        )
        return result == "passed"

    def evaluate_sample_io(self, item: dict, cur_imp: str, language: str):
        setup, test_list = self._decode_item(item)
        sample_io = test_list[:2] if isinstance(test_list, list) else []
        
        sample_io_with_setup = [setup + "\n" + io for io in sample_io]

        return evaluate_io(sample_io=sample_io_with_setup, completion=cur_imp)
    
    def evaluate_additional_io(self, id: int, io: List[str], cur_imp: str, language: str):
        if len(io) == 0:
            return True, ""
        return evaluate_io(sample_io=io, completion=cur_imp)

    @staticmethod
    def get_prompt(item):
        if "instruction" in item:
            base = item['instruction'].strip()
        elif "prompt" in item:
            base = item['prompt'].strip()
        elif "text" in item:
            base = item['text'].strip()
        else:
            raise Exception("No prompt, instruction, or text found in the item")

        if item.get('signature'):
            base += f"\n\nThe function signature is:\n```python\n{item['signature'].strip()}\n```"

        return base
