import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from docker_utils import run_in_sandbox

from typing import *
import contextlib
import signal

from .executor_utils import function_with_timeout

def evaluate_io(
    sample_io: list[str],
    completion: str,
    timeout: int = 5,
    stop_early: bool = False,
):
    if len(sample_io) == 0:
        return True, ""

    test_log = ""
    passed = True
    ai_code = ("from typing import *\n" if "from typing import *" not in completion else "") + completion
    for io in sample_io:
        test_cases_code = io + "\n"
        is_passed, log = run_in_sandbox(ai_code=ai_code, test_cases=test_cases_code, timeout=timeout)
        
        if is_passed:
            test_log += f"Passed in test case: {io}\n"
        else:
            if stop_early:
                return False, f"Failed in test case: {io}\nLog: {log}\n"
            passed = False
            test_log += f"Failed in test case: {io}\nLog: {log}\n"

    return passed, test_log


def evaluate_io_et(
    sample_io: list[str],
    completion: str,
    timeout: int = 5,
    prompt: str = "",
):
    io = "\n".join(sample_io)
    
    ai_code = ("from typing import *\n" if "from typing import *" not in completion else "") + prompt + completion
    test_cases_code = io + "\n"
    
    passed, log = run_in_sandbox(ai_code=ai_code, test_cases=test_cases_code, timeout=timeout)
    
    if passed:
        return "passed"
    else:
        return f"failed: {log}"


def evaluate_functional_correctness(
    test: str,
    entry_point: str,
    completion: str,
    timeout: int = 5,
):
    ai_code = ("from typing import *\n" if "from typing import *" not in completion else "") + completion
    test_cases_code = test
    if entry_point:
        test_cases_code += "\n" + f"check({entry_point})"

    passed, log = run_in_sandbox(ai_code=ai_code, test_cases=test_cases_code, timeout=timeout)
    
    if passed:
        return "passed"
    else:
        return f"failed: {log}"


class TimeoutException(Exception):
    pass
