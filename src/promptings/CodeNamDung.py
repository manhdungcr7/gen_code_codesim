"""
CodeNamDung — CodeSIM + Enhanced Fault-Localisation Debugging

Architecture (giữ nguyên CodeSIM, thêm 1 lớp debug mới):

    for plan_no in 1..p:
        [Planning Agent]   — giống CodeSIM
        [Coding Agent]     — giống CodeSIM
        if passed: break

        [Standard Debug]   — giống CodeSIM (d lần)
        if passed: break

        [Enhanced Debug]   — MỚI: fault localisation + targeted fix (e lần)
        if passed: break

Enhanced Debug khác Standard Debug ở chỗ:
  - Truyền toàn bộ lịch sử các lần debug đã thất bại
  - Yêu cầu LLM xác định ĐÚNG DÒNG / ĐIỀU KIỆN bị lỗi trước
  - Chỉ sửa tối thiểu (minimal fix), không rewrite toàn bộ
  - Yêu cầu trace thủ công để verify trước khi submit
"""

import re

from .CodeSIM import (
    CodeSIM,
    prompt_for_planning,
    prompt_for_simulation,
    prompt_for_plan_refinement,
    prompt_for_code_generation,
    prompt_for_debugging,
)
from utils.parse import parse_response
from constants.verboseType import VERBOSE_FULL


# ─── Enhanced Debug Prompt ──────────────────────────────────────────────────────
prompt_for_enhanced_debugging = """\
You are a senior programmer debugging a solution that has already failed \
{total_prev_attempts} previous fix attempts. Those attempts all used the same \
"simulate then fix" strategy but still produced wrong code.

{problem_with_planning}

### Current Buggy Code
```{language}
{code}
```

{test_log}

### History of Failed Fix Attempts
{debug_history_str}

**Expected Output:**

Your response must be structured as follows:

### Fault Localisation
- Review the current code and ALL previous failed attempts carefully.
- Identify the SINGLE most suspicious line or expression. Be specific:
  e.g. "Line 7: `if x > threshold` — should be `>=`" or
       "The variable `count` is never reset between outer loop iterations".
- Do NOT say "the plan might be wrong" unless you can point to exactly which
  step of the plan is incorrect and why.

### Root Cause Hypothesis
- State ONE precise, falsifiable hypothesis about the bug.
  Good: "The slice `arr[lo:hi]` excludes `hi`, so the last element is missed."
  Bad: "The algorithm might not handle edge cases."

### Minimal Fix
- Change ONLY the identified fault location.
- Do NOT restructure, refactor, or rewrite other parts.

### Verification
- Manually trace the FIXED code on the failing test case, step by step.
- Show intermediate variable values.
- Confirm the output now equals the expected output.

### Modified Code
```{language}
# Minimal targeted fix — only the identified fault location is changed.
```

--------
**Important Instructions:**
- Fix exactly ONE thing. If you find multiple bugs, fix only the most critical one.
- Do not add assert or print statements.
- The modified **{language}** code must be enclosed in triple backticks (```).
- Your response must contain **Fault Localisation**, **Root Cause Hypothesis**,
  **Minimal Fix**, **Verification**, and **Modified Code** sections.
{std_input_prompt}"""


# ─── Helper ────────────────────────────────────────────────────────────────────

def _extract_debugging_notes(response: str) -> str:
    """Pull the 'Debugging Notes' / 'Root Cause' section from an LLM response."""
    for header in ("### Debugging Notes", "### Root Cause Hypothesis",
                   "### Fault Localisation"):
        if header in response:
            snippet = response[response.find(header):]
            # Take up to 600 chars so the history doesn't blow up the context
            return snippet[:600].strip()
    # Fallback: first 400 chars of response
    return response[:400].strip()


def _format_debug_history(history: list[dict]) -> str:
    """Format debug history list into a concise string for the prompt."""
    if not history:
        return "(no previous attempts)"
    lines = []
    for i, entry in enumerate(history, 1):
        lines.append(f"#### Attempt {i}")
        lines.append(f"**LLM diagnosis / fix tried:**\n{entry['notes']}")
        lines.append(f"**Result:** still failed — {entry['test_summary']}\n")
    return "\n".join(lines)


def _test_summary(test_log: str, max_chars: int = 200) -> str:
    """Return a short summary of a test_log for the history."""
    if not test_log:
        return "no test log available"
    # Keep only first line with "assert" or the first max_chars chars
    for line in test_log.splitlines():
        if "assert" in line.lower() or "error" in line.lower():
            return line.strip()[:max_chars]
    return test_log[:max_chars].strip()


# ─── CodeNamDung ───────────────────────────────────────────────────────────────

class CodeNamDung(CodeSIM):
    """
    CodeSIM + Enhanced Fault-Localisation Debug module.

    Parameters
    ----------
    max_enhanced_debug_try : int
        Number of additional enhanced-debug attempts after the standard
        CodeSIM debug loop exhausts its budget.  Default = 3.
    All other parameters are forwarded to CodeSIM unchanged.
    """

    def __init__(
        self,
        max_enhanced_debug_try: int = 3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_enhanced_debug_try = max_enhanced_debug_try

        if self.verbose >= VERBOSE_FULL:
            print(
                f"[CodeNamDung] max_enhanced_debug_try={self.max_enhanced_debug_try}",
                flush=True,
            )

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

        additional_io = []
        self.run_details["additional_io"] = additional_io

        # ── Outer: Planning loop ───────────────────────────────────────────────
        for plan_no in range(1, self.max_plan_try + 1):

            # ── Planning phase (identical to CodeSIM) ─────────────────────────
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
                print(f"[CodeNamDung] Input for Planning: {plan_no}\n")
                print(input_for_planning[0]["content"], flush=True)

            response = self.gpt_chat(processed_input=input_for_planning)

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(f"[CodeNamDung] Response from Planning: {plan_no}\n")
                print(response, flush=True)

            plan = (
                response[response.rfind("### Plan"):]
                if "### Plan" in response
                else f"### Plan\n\n{response}"
            )
            problem_with_planning = f"## Problem:\n{problem}\n\n{plan}"

            # ── Simulation / Plan Validation (identical to CodeSIM) ───────────
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
                print(f"[CodeNamDung] Input for Simulation: {plan_no}\n")
                print(input_for_simulation[0]["content"], flush=True)

            response = self.gpt_chat(processed_input=input_for_simulation)

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(f"[CodeNamDung] Response from Simulation: {plan_no}\n")
                print(response, flush=True)

            if (
                "Plan Modification Needed" in response
                and "No Plan Modification Needed" not in response
            ):
                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print("[CodeNamDung] Plan Modification Needed.\n")

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
                    print(f"[CodeNamDung] Input for Plan Refinement: {plan_no}\n")
                    print(input_for_plan_refinement[0]["content"], flush=True)

                plan = self.gpt_chat(processed_input=input_for_plan_refinement)

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(f"[CodeNamDung] Response from Plan Refinement: {plan_no}\n")
                    print(plan, flush=True)

                problem_with_planning = f"## Problem:\n{problem}\n\n{plan}"

            # ── Code Generation (identical to CodeSIM) ────────────────────────
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
                print("[CodeNamDung] Input for Code Generation:\n")
                print(input_for_code_gen[0]["content"], flush=True)

            response = self.gpt_chat(processed_input=input_for_code_gen)

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print("[CodeNamDung] Response from Code Generation:\n")
                print(response, flush=True)

            code = parse_response(response)
            passed, test_log = self.check(data_row, additional_io, code)

            if passed:
                break

            # ── Standard Debug loop (identical to CodeSIM) ────────────────────
            # Accumulate history for the enhanced debug phase below.
            debug_history: list[dict] = []

            for debug_no in range(1, self.max_debug_try + 1):
                input_for_debugging = [
                    {
                        "role": "user",
                        "content": prompt_for_debugging.format(
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
                        f"[CodeNamDung] Input for Standard Debug: {plan_no}, {debug_no}\n"
                    )
                    print(input_for_debugging[0]["content"], flush=True)

                response = self.gpt_chat(processed_input=input_for_debugging)

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(
                        f"[CodeNamDung] Response from Standard Debug: "
                        f"{plan_no}, {debug_no}\n"
                    )
                    print(response, flush=True)

                # Save this attempt to history before updating code/test_log
                debug_history.append(
                    {
                        "notes": _extract_debugging_notes(response),
                        "test_summary": _test_summary(test_log),
                    }
                )

                code = parse_response(response)
                passed, test_log = self.check(data_row, additional_io, code)

                if passed:
                    break

            if passed:
                break

            # ── Enhanced Fault-Localisation Debug loop (CodeNamDung NEW) ──────
            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "=" * 70)
                print(
                    f"[CodeNamDung] Standard debug exhausted after "
                    f"{self.max_debug_try} attempts. "
                    f"Switching to Enhanced Fault-Localisation Debug."
                )
                print("=" * 70, flush=True)

            for enh_no in range(1, self.max_enhanced_debug_try + 1):
                history_str = _format_debug_history(debug_history)

                input_for_enhanced = [
                    {
                        "role": "user",
                        "content": prompt_for_enhanced_debugging.format(
                            total_prev_attempts=len(debug_history),
                            problem_with_planning=problem_with_planning,
                            code=code,
                            language=self.language,
                            test_log=test_log,
                            debug_history_str=history_str,
                            std_input_prompt=std_input_prompt,
                        ),
                    }
                ]
                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(
                        f"[CodeNamDung] Input for Enhanced Debug: "
                        f"{plan_no}, {enh_no}\n"
                    )
                    print(input_for_enhanced[0]["content"], flush=True)

                response = self.gpt_chat(processed_input=input_for_enhanced)

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(
                        f"[CodeNamDung] Response from Enhanced Debug: "
                        f"{plan_no}, {enh_no}\n"
                    )
                    print(response, flush=True)

                # Append this enhanced attempt to history too
                debug_history.append(
                    {
                        "notes": _extract_debugging_notes(response),
                        "test_summary": _test_summary(test_log),
                    }
                )

                code = parse_response(response)
                passed, test_log = self.check(data_row, additional_io, code)

                if passed:
                    if self.verbose >= VERBOSE_FULL:
                        print(
                            f"\n[CodeNamDung] Enhanced Debug fixed the bug "
                            f"on attempt {enh_no}!",
                            flush=True,
                        )
                    break

            if passed:
                break

        if self.verbose >= VERBOSE_FULL:
            print("\n\n" + "_" * 70)

        return code
