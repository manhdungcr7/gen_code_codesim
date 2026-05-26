"""
CodeNamDungV3 — CodeSIM + Error-Classified Debugging

Architecture:

    for plan_no in 1..p:
        [Planning]          — same as CodeSIM
        [Simulation]        — same as CodeSIM
        [Coding]            — same as CodeSIM
        if passed: break

        for debug_no in 1..d:
            [Error Classifier]  — NEW: inspect test_log each iteration
              → RUNTIME  : TypeError, IndexError, NameError, etc.
              → LOGIC     : AssertionError (code runs but wrong output)

            [Runtime Debug]     — NEW: traceback-focused, minimal targeted fix
            or
            [Logic Debug]       — NEW: step-by-step variable trace to find
                                        exact divergence line

        if passed: break

Why this is better than a single generic debug prompt:
  - Runtime errors have a known location (traceback line) and known fix pattern
    (type coercion, bounds check, None guard).  Generic prompts waste tokens
    re-simulating logic that is not the problem.
  - Logic errors need step-by-step variable tracing to pinpoint the divergence.
    Generic prompts produce "I think the bug is in X" without rigorous tracing,
    leading to repeated incorrect fixes.
  - Error type can CHANGE between iterations (fixing a runtime error may expose
    a logic error), so classification runs before every debug attempt.
"""

from typing import List

from .CodeSIM import (
    CodeSIM,
    prompt_for_planning,
    prompt_for_simulation,
    prompt_for_plan_refinement,
    prompt_for_code_generation,
)
from utils.parse import parse_response
from constants.verboseType import VERBOSE_FULL


# ─── Error Classification ────────────────────────────────────────────────────────

_RUNTIME_ERROR_TYPES = [
    "TypeError", "IndexError", "AttributeError", "NameError",
    "ValueError", "KeyError", "ZeroDivisionError", "StopIteration",
    "RecursionError", "OverflowError", "SyntaxError", "IndentationError",
    "UnboundLocalError", "RuntimeError",
]


def _classify_error(test_log: str) -> str:
    """Return 'runtime' if a traceback error is present, else 'logic'."""
    for err in _RUNTIME_ERROR_TYPES:
        if err in test_log:
            return "runtime"
    return "logic"


# ─── Runtime Error Debug Prompt ─────────────────────────────────────────────────

prompt_for_runtime_debug = """\
Your code crashes with a runtime error. Focus only on fixing that specific error.

{problem_with_planning}

### Buggy Code
```{language}
{code}
```

### Error Log
{test_log}

**Your task:**

### Error Analysis
- Quote the EXACT line from the traceback where the crash occurs.
- State the error type and message verbatim.
- State what value or type the variable/expression actually has at that point.
- State what value or type the code expects at that point.

### Root Cause
One sentence only:
e.g. "`parent[(x, y)]` is `None` at the root node, but the code tries to \
unpack it as a tuple `(px, py) = parent[(x, y)]`."

### Minimal Fix
- Change ONLY the line(s) identified above.
- Do NOT restructure or rewrite other parts of the code.
- Do NOT add assert or print statements.

### Modified Code
```{language}
# Only the crash site is changed.
```

--------
**Important Instructions:**
- The modified **{language}** code must be enclosed in triple backticks (```).
- Fix exactly the crash site. Do not change unrelated logic.
- Your response must contain **Error Analysis**, **Root Cause**, **Minimal Fix**, \
and **Modified Code** sections.
{std_input_prompt}"""


# ─── Logic / Wrong-Answer Debug Prompt ──────────────────────────────────────────

prompt_for_logic_debug = """\
Your code runs without crashing but produces wrong output. The bug is in the logic.

{problem_with_planning}

### Buggy Code
```{language}
{code}
```

### Failing Tests
{test_log}

**Your task:**

### Step-by-Step Variable Trace
- Take the FIRST failing test case.
- Execute the code line by line in your head, tracking the value of every \
key variable after each statement.
- Use this format for each step:
    line <N>: <variable> = <actual_value>
- Continue until you find the FIRST line where a variable holds a wrong value.
- Mark that line explicitly:
    line <N>: <variable> = <actual_value>  ← WRONG, expected <correct_value>

Do not skip steps. Show every intermediate value.

### Root Cause
One sentence: what is the exact logical error?
e.g. "The loop uses `<` instead of `<=`, so the last element is never processed."

### Fix
- Change ONLY the line(s) identified above.
- Do NOT restructure or rewrite other parts.
- Do NOT add assert or print statements.

### Modified Code
```{language}
# Only the identified logical error is corrected.
```

--------
**Important Instructions:**
- The modified **{language}** code must be enclosed in triple backticks (```).
- Fix exactly the identified line. Do not change unrelated logic.
- Your response must contain **Step-by-Step Variable Trace**, **Root Cause**, \
**Fix**, and **Modified Code** sections.
{std_input_prompt}"""


# ─── CodeNamDungV3 ───────────────────────────────────────────────────────────────

class CodeNamDungV3(CodeSIM):
    """
    CodeSIM + Error-Classified Debugging.

    Each debug attempt first classifies the current failure as a runtime error
    (has a traceback) or a logic/wrong-answer error (AssertionError), then
    applies a debug prompt that is appropriate for that error type.

    Parameters
    ----------
    All parameters forwarded to CodeSIM unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.verbose >= VERBOSE_FULL:
            print("[CodeNamDungV3] Error-Classified Debugging enabled.", flush=True)

    # ── Override run_single_pass ───────────────────────────────────────────────
    def run_single_pass(self, data_row: dict):
        print("", flush=True)

        problem = self.data.get_prompt(data_row)

        std_input_prompt = ""
        if self.is_competitive:
            std_input_prompt = (
                "- Strictly follow the sample input and output format.\n"
                "    - The input should be taken from Standard input and output "
                "should be given to standard output. If you are writing a function "
                "then after the function definition take the input using `input()` "
                "function then call the function with specified parameters and "
                "finally print the output of the function.\n"
                "    - For array input parse the array then pass it to the function."
                " Parsing technique is given in the sample input output format "
                "section.\n"
                "    - Do not add extra print statement otherwise it will failed "
                "the test cases."
            )
            problem = problem[: problem.find("-------\nImportant Note:")]

        additional_io: List[str] = []
        self.run_details["additional_io"] = additional_io

        # ── Outer: Planning loop ───────────────────────────────────────────────
        for plan_no in range(1, self.max_plan_try + 1):

            # ── Planning phase ─────────────────────────────────────────────────
            input_for_planning = [
                {
                    "role": "user",
                    "content": prompt_for_planning.format(
                        problem=problem,
                        language=self.language,
                    ),
                }
            ]
            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(f"[CodeNamDungV3] Input for Planning: {plan_no}\n")
                print(input_for_planning[0]["content"], flush=True)

            response = self.gpt_chat(processed_input=input_for_planning)

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(f"[CodeNamDungV3] Response from Planning: {plan_no}\n")
                print(response, flush=True)

            plan = (
                response[response.rfind("### Plan"):]
                if "### Plan" in response
                else f"### Plan\n\n{response}"
            )
            problem_with_planning = f"## Problem:\n{problem}\n\n{plan}"

            # ── Simulation / Plan Validation ───────────────────────────────────
            input_for_simulation = [
                {
                    "role": "user",
                    "content": prompt_for_simulation.format(
                        problem_with_planning=problem_with_planning,
                        language=self.language,
                    ),
                }
            ]
            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(f"[CodeNamDungV3] Input for Simulation: {plan_no}\n")
                print(input_for_simulation[0]["content"], flush=True)

            response = self.gpt_chat(processed_input=input_for_simulation)

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(f"[CodeNamDungV3] Response from Simulation: {plan_no}\n")
                print(response, flush=True)

            if (
                "Plan Modification Needed" in response
                and "No Plan Modification Needed" not in response
            ):
                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print("[CodeNamDungV3] Plan Modification Needed.\n")

                input_for_plan_refinement = [
                    {
                        "role": "user",
                        "content": prompt_for_plan_refinement.format(
                            problem_with_planning=problem_with_planning,
                            language=self.language,
                            critique=response,
                        ),
                    }
                ]
                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(f"[CodeNamDungV3] Input for Plan Refinement: {plan_no}\n")
                    print(input_for_plan_refinement[0]["content"], flush=True)

                plan = self.gpt_chat(processed_input=input_for_plan_refinement)

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(
                        f"[CodeNamDungV3] Response from Plan Refinement: {plan_no}\n"
                    )
                    print(plan, flush=True)

                problem_with_planning = f"## Problem:\n{problem}\n\n{plan}"

            # ── Code Generation ────────────────────────────────────────────────
            input_for_code_gen = [
                {
                    "role": "user",
                    "content": prompt_for_code_generation.format(
                        problem_with_planning=problem_with_planning,
                        language=self.language,
                        std_input_prompt=std_input_prompt,
                    ),
                }
            ]
            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print("[CodeNamDungV3] Input for Code Generation:\n")
                print(input_for_code_gen[0]["content"], flush=True)

            response = self.gpt_chat(processed_input=input_for_code_gen)

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print("[CodeNamDungV3] Response from Code Generation:\n")
                print(response, flush=True)

            code = parse_response(response)
            passed, test_log = self.check(data_row, additional_io, code)

            if passed:
                break

            # ── Error-Classified Debug loop ────────────────────────────────────
            for debug_no in range(1, self.max_debug_try + 1):

                error_type = _classify_error(test_log)

                if error_type == "runtime":
                    debug_prompt = prompt_for_runtime_debug
                    prompt_label = "Runtime Debug"
                else:
                    debug_prompt = prompt_for_logic_debug
                    prompt_label = "Logic Debug"

                input_for_debug = [
                    {
                        "role": "user",
                        "content": debug_prompt.format(
                            problem_with_planning=problem_with_planning,
                            code=code,
                            language=self.language,
                            test_log=test_log,
                            std_input_prompt=std_input_prompt,
                        ),
                    }
                ]
                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(
                        f"[CodeNamDungV3] Input for {prompt_label}: "
                        f"{plan_no}, {debug_no} "
                        f"[error_type={error_type}]\n"
                    )
                    print(input_for_debug[0]["content"], flush=True)

                response = self.gpt_chat(processed_input=input_for_debug)

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(
                        f"[CodeNamDungV3] Response from {prompt_label}: "
                        f"{plan_no}, {debug_no}\n"
                    )
                    print(response, flush=True)

                code = parse_response(response)
                passed, test_log = self.check(data_row, additional_io, code)

                if passed:
                    break

            if passed:
                break

        if self.verbose >= VERBOSE_FULL:
            print("\n\n" + "_" * 70)

        return code
