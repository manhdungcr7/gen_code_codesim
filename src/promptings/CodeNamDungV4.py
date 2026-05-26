"""
CodeNamDungV4 — CodeSIM + Test-Failure-Driven Plan Revision

Architecture:

    for plan_no in 1..p:
        [Planning]          — same as CodeSIM
        [Simulation]        — same as CodeSIM (uses problem examples)
        [Coding]            — same as CodeSIM
        if passed: break

        [Plan Revision]     — NEW: trace plan on actual failing test,
                               detect PLAN_ERROR vs CODE_ERROR,
                               revise plan + recode if PLAN_ERROR
        if passed: break

        [Standard Debug]    — same as CodeSIM
        if passed: break

Why this is better than standard re-planning:
  - CodeSIM's Simulation uses only sample examples from problem text.
    Hidden test failures never feed back into plan correction.
  - Plan Revision uses the ACTUAL failing test as evidence, so the LLM
    can see exactly where its plan's output diverges from the assertion.
  - Unlike test-driven re-planning (V2), the problem text is kept — only
    the failing test is added as extra signal. This avoids the risk of
    LLM ignoring the problem semantics entirely.
  - Unlike debug (which patches code), this targets the root cause when
    the plan itself is logically wrong.
"""

from typing import List

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


# ─── Plan Revision Prompt ────────────────────────────────────────────────────────

prompt_for_plan_revision = """\
Your code failed on the test case below. Before debugging the code, \
check whether the PLAN itself has a logical flaw.

{problem_with_planning}

### Current Code
```{language}
{code}
```

### First Failing Test
{test_log}

**Your task — Plan Revision Check. Follow all four steps:**

### Step 1: Plan Trace on Failing Input
- Extract the INPUT from the first failing test case above.
- Apply your plan step by step to that input.
- Show intermediate values at each step.
- State clearly: what does your plan produce as output?

### Step 2: Compare with Test Evidence
Fill in both lines:
  - My plan produces: [your answer from Step 1]
  - Test asserts:     [the expected value from the assertion]
  - Same? Yes / No

### Step 3: Verdict
Write exactly ONE verdict on its own line:
  **PLAN_ERROR** — Step 2 says No: the plan produces the wrong result
  **CODE_ERROR**  — Step 2 says Yes: the plan is correct, only the code is wrong

### Step 4: Revised Plan (write ONLY if PLAN_ERROR)
  - Identify the exact step in the current plan that is wrong.
  - State what it should do instead, derived from the test evidence.
  - Write a complete corrected plan.
  - Verify: mentally apply the revised plan to the same input and confirm \
it now produces the correct output from Step 2.

--------
Important instructions:
- Do NOT generate code.
- Step 1 must show actual intermediate values, not vague descriptions.
- If PLAN_ERROR, the revised plan must demonstrably fix the exact issue found in Step 2.
{std_input_prompt}"""


# ─── Helpers ─────────────────────────────────────────────────────────────────────

def _is_plan_error(response: str) -> bool:
    """Return True if the revision response declares PLAN_ERROR."""
    for line in response.splitlines():
        s = line.strip()
        if "**PLAN_ERROR**" in s or s == "PLAN_ERROR":
            return True
        if "**CODE_ERROR**" in s or s == "CODE_ERROR":
            return False
    return False


def _extract_revised_plan(response: str) -> str | None:
    """Extract the Revised Plan section from a PLAN_ERROR response."""
    for header in ("### Step 4: Revised Plan", "### Revised Plan",
                   "## New Plan", "### New Plan"):
        if header in response:
            return response[response.find(header):]
    # Fallback: content after PLAN_ERROR verdict that contains a plan
    verdict_pos = response.find("PLAN_ERROR")
    if verdict_pos != -1:
        after = response[verdict_pos:]
        for kw in ("### Plan", "## Plan", "1."):
            if kw in after:
                return after[after.find(kw):]
    return None


# ─── CodeNamDungV4 ───────────────────────────────────────────────────────────────

class CodeNamDungV4(CodeSIM):
    """
    CodeSIM + Test-Failure-Driven Plan Revision.

    After every code generation failure, the plan is traced on the actual
    failing test to determine whether the root cause is a plan-level error
    (PLAN_ERROR) or a code-level error (CODE_ERROR).

    - PLAN_ERROR: a revised plan is extracted and a new code is generated
      from it immediately (within the same plan iteration), skipping straight
      to code generation without re-running simulation.
    - CODE_ERROR: plan revision is skipped; standard debug proceeds as in
      CodeSIM.

    Parameters
    ----------
    max_revision_try : int
        How many plan revision attempts per plan iteration.  Default = 1.
    All other parameters forwarded to CodeSIM unchanged.
    """

    def __init__(self, max_revision_try: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_revision_try = max_revision_try

        if self.verbose >= VERBOSE_FULL:
            print(
                f"[CodeNamDungV4] max_revision_try={self.max_revision_try}",
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

        additional_io: List[str] = []
        self.run_details["additional_io"] = additional_io

        # ── Outer: Planning loop ───────────────────────────────────────────────
        for plan_no in range(1, self.max_plan_try + 1):

            # ── Planning ───────────────────────────────────────────────────────
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
                print(f"[CodeNamDungV4] Input for Planning: {plan_no}\n")
                print(input_for_planning[0]["content"], flush=True)

            response = self.gpt_chat(processed_input=input_for_planning)

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(f"[CodeNamDungV4] Response from Planning: {plan_no}\n")
                print(response, flush=True)

            plan = (
                response[response.rfind("### Plan"):]
                if "### Plan" in response
                else f"### Plan\n\n{response}"
            )
            problem_with_planning = f"## Problem:\n{problem}\n\n{plan}"

            # ── Simulation ─────────────────────────────────────────────────────
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
                print(f"[CodeNamDungV4] Input for Simulation: {plan_no}\n")
                print(input_for_simulation[0]["content"], flush=True)

            response = self.gpt_chat(processed_input=input_for_simulation)

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(f"[CodeNamDungV4] Response from Simulation: {plan_no}\n")
                print(response, flush=True)

            if (
                "Plan Modification Needed" in response
                and "No Plan Modification Needed" not in response
            ):
                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print("[CodeNamDungV4] Plan Modification Needed.\n")

                input_for_refinement = [
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
                    print(
                        f"[CodeNamDungV4] Input for Plan Refinement: {plan_no}\n"
                    )
                    print(input_for_refinement[0]["content"], flush=True)

                plan = self.gpt_chat(processed_input=input_for_refinement)

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(
                        f"[CodeNamDungV4] Response from Plan Refinement: {plan_no}\n"
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
                print("[CodeNamDungV4] Input for Code Generation:\n")
                print(input_for_code_gen[0]["content"], flush=True)

            response = self.gpt_chat(processed_input=input_for_code_gen)

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print("[CodeNamDungV4] Response from Code Generation:\n")
                print(response, flush=True)

            code = parse_response(response)
            passed, test_log = self.check(data_row, additional_io, code)

            if passed:
                break

            # ── Plan Revision from Failing Test ────────────────────────────────
            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "=" * 70)
                print(
                    f"[CodeNamDungV4] Code failed. Running Plan Revision "
                    f"(plan_no={plan_no})."
                )
                print("=" * 70, flush=True)

            revision_fixed = False

            for rev_no in range(1, self.max_revision_try + 1):
                input_for_revision = [
                    {
                        "role": "user",
                        "content": prompt_for_plan_revision.format(
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
                        f"[CodeNamDungV4] Input for Plan Revision: "
                        f"{plan_no}, {rev_no}\n"
                    )
                    print(input_for_revision[0]["content"], flush=True)

                rev_response = self.gpt_chat(processed_input=input_for_revision)

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(
                        f"[CodeNamDungV4] Response from Plan Revision: "
                        f"{plan_no}, {rev_no}\n"
                    )
                    print(rev_response, flush=True)

                if _is_plan_error(rev_response):
                    revised_plan = _extract_revised_plan(rev_response)
                    if revised_plan:
                        if self.verbose >= VERBOSE_FULL:
                            print(
                                f"\n[CodeNamDungV4] PLAN_ERROR detected. "
                                f"Regenerating code from revised plan.",
                                flush=True,
                            )

                        # Use revised plan — skip simulation (test evidence is
                        # stronger than problem-example simulation here)
                        problem_with_planning = (
                            f"## Problem:\n{problem}\n\n{revised_plan}"
                        )

                        input_for_recode = [
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
                            print(
                                "[CodeNamDungV4] Input for Re-coding "
                                f"(revised plan): {plan_no}, {rev_no}\n"
                            )
                            print(input_for_recode[0]["content"], flush=True)

                        recode_response = self.gpt_chat(
                            processed_input=input_for_recode
                        )

                        if self.verbose >= VERBOSE_FULL:
                            print("\n\n" + "_" * 70)
                            print(
                                "[CodeNamDungV4] Response from Re-coding "
                                f"(revised plan): {plan_no}, {rev_no}\n"
                            )
                            print(recode_response, flush=True)

                        code = parse_response(recode_response)
                        passed, test_log = self.check(
                            data_row, additional_io, code
                        )

                        if passed:
                            revision_fixed = True
                            if self.verbose >= VERBOSE_FULL:
                                print(
                                    f"\n[CodeNamDungV4] Revised plan fixed the "
                                    f"problem on revision {rev_no}!",
                                    flush=True,
                                )
                            break
                    else:
                        if self.verbose >= VERBOSE_FULL:
                            print(
                                f"\n[CodeNamDungV4] PLAN_ERROR declared but "
                                f"no revised plan extracted. "
                                f"Proceeding to standard debug.",
                                flush=True,
                            )
                else:
                    if self.verbose >= VERBOSE_FULL:
                        print(
                            f"\n[CodeNamDungV4] CODE_ERROR on revision {rev_no}. "
                            f"Plan is correct — proceeding to standard debug.",
                            flush=True,
                        )

            if revision_fixed:
                break

            # ── Standard Debug loop ────────────────────────────────────────────
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
                        f"[CodeNamDungV4] Input for Standard Debug: "
                        f"{plan_no}, {debug_no}\n"
                    )
                    print(input_for_debugging[0]["content"], flush=True)

                response = self.gpt_chat(processed_input=input_for_debugging)

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(
                        f"[CodeNamDungV4] Response from Standard Debug: "
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
