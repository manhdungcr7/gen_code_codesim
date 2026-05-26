"""
CodeNamDungV2 — CodeNamDung + Test-Driven Re-planning

Architecture (build on CodeNamDung, thêm 1 lớp re-planning mới):

    for plan_no in 1..p:
        [Planning Agent]           — giống CodeSIM (dùng test_derived_plan nếu có)
        [Simulation Agent]         — giống CodeSIM
        [Coding Agent]             — giống CodeSIM
        if passed: break

        [Standard Debug]           — giống CodeSIM (d lần)
        if passed: break

        [Test-Driven Re-planning]  — MỚI: bỏ problem text, re-derive spec từ test assertions
          → sinh new_plan cho plan_no tiếp theo
          → nếu không extract được plan: fallback sang enhanced debug

        [Enhanced Debug]           — giống CodeNamDung (e lần, chỉ khi fallback)
        if passed: break

Test-Driven Re-planning khác Standard/Enhanced Debug ở chỗ:
  - Không cho LLM đọc problem text (nguồn gốc bias)
  - Chỉ dùng test assertions làm ground truth
  - LLM tự derive spec: target cell, return format, edge cases — từ assertions
  - Sinh plan mới hoàn toàn, không patch plan cũ
"""

from .CodeNamDung import (
    CodeNamDung,
    prompt_for_enhanced_debugging,
    _extract_debugging_notes,
    _format_debug_history,
    _test_summary,
)
from .CodeSIM import (
    prompt_for_planning,
    prompt_for_simulation,
    prompt_for_plan_refinement,
    prompt_for_code_generation,
    prompt_for_debugging,
)
from utils.parse import parse_response
from constants.verboseType import VERBOSE_FULL


# ─── Test-Driven Re-planning Prompt ─────────────────────────────────────────────

prompt_for_test_driven_replanning = """\
Your code has failed {num_debug} debug attempts without being fixed. \
The test assertions below are the ground truth — they define exactly \
what the function must do.

### Test Assertions (ground truth)
{test_log}

Do NOT read the problem description. Derive the function's full specification \
ONLY from the assertions above.

---

### Step 1: Derive the Specification
For each test assertion, state concretely what input → output the function must produce.
Be explicit about:
  - The exact return type and structure (e.g. "(bool, list of tuples)")
  - Every key value that the assertions reveal (e.g. which cell is the exit/goal, \
what constitutes success vs failure, what the path must contain)

### Step 2: Gap Analysis
Compare what your CURRENT plan produces against what Step 1 requires.
State the EXACT difference with concrete values — not vague descriptions:
  e.g. "My plan targets cell (2,0) but test assertions show the path ends at (2,2)."
  e.g. "My plan returns only a bool but tests expect (bool, list of tuples)."
If there is no gap (your plan is consistent with all assertions), state that explicitly.

### Step 3: New Plan
Write a complete, step-by-step plan that satisfies ALL assertions from Step 1.
  - Derive ALL assumptions (target cell, return format, edge cases) ONLY from \
the test assertions — NOT from the problem description.
  - Every assumption must be traceable to a specific assertion.
  - The plan MUST differ from the current plan in the exact point identified in Step 2.

--------
Important instructions:
- Do NOT generate code.
- Do NOT reference the problem description anywhere.
- Every claim in Step 2 must be a concrete value, not a vague description.
{std_input_prompt}"""


# ─── Helpers ─────────────────────────────────────────────────────────────────────

def _extract_new_plan(response: str) -> str | None:
    """Extract the New Plan section from a test-driven re-planning response."""
    for header in ("### Step 3: New Plan", "### New Plan", "### Step 3"):
        if header in response:
            return response[response.find(header):]
    if "### Plan" in response:
        return response[response.rfind("### Plan"):]
    return None


# ─── CodeNamDungV2 ───────────────────────────────────────────────────────────────

class CodeNamDungV2(CodeNamDung):
    """
    CodeNamDung + Test-Driven Re-planning.

    After standard debug is exhausted, instead of re-reading the (potentially
    misleading) problem description, a new plan is derived SOLELY from the
    failing test assertions.  This breaks the anchoring bias where the LLM
    keeps trusting the problem text over the test evidence.

    Parameters
    ----------
    max_replan_try : int
        How many test-driven re-planning attempts per planning iteration.
        Default = 1.
    max_enhanced_debug_try : int
        Fallback enhanced debug attempts if re-planning fails to extract a plan.
        Default = 0 (disabled; re-planning is the primary recovery mechanism).
    All other parameters forwarded to CodeNamDung / CodeSIM unchanged.
    """

    def __init__(
        self,
        max_replan_try: int = 1,
        max_enhanced_debug_try: int = 0,
        *args,
        **kwargs,
    ):
        super().__init__(max_enhanced_debug_try=max_enhanced_debug_try, *args, **kwargs)
        self.max_replan_try = max_replan_try

        if self.verbose >= VERBOSE_FULL:
            print(
                f"[CodeNamDungV2] "
                f"max_replan_try={self.max_replan_try}, "
                f"max_enhanced_debug_try={self.max_enhanced_debug_try}",
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

        # Carries a test-derived plan into the next planning iteration
        test_derived_plan: str | None = None

        # ── Outer: Planning loop ───────────────────────────────────────────────
        for plan_no in range(1, self.max_plan_try + 1):

            # ── Planning phase ─────────────────────────────────────────────────
            skip_simulation = False
            if test_derived_plan is not None:
                # Use the test-derived plan directly — skip LLM planning call
                plan = test_derived_plan
                test_derived_plan = None
                skip_simulation = True  # simulation would contradict test-derived plan
                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "=" * 70)
                    print(
                        f"[CodeNamDungV2] Using test-derived plan for "
                        f"planning iteration {plan_no}."
                    )
                    print("=" * 70, flush=True)
            else:
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
                    print(f"[CodeNamDungV2] Input for Planning: {plan_no}\n")
                    print(input_for_planning[0]["content"], flush=True)

                response = self.gpt_chat(processed_input=input_for_planning)

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(f"[CodeNamDungV2] Response from Planning: {plan_no}\n")
                    print(response, flush=True)

                plan = (
                    response[response.rfind("### Plan"):]
                    if "### Plan" in response
                    else f"### Plan\n\n{response}"
                )

            if skip_simulation:
                # Plan was derived from test assertions; trust it as-is.
                problem_with_planning = f"{plan}"
                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(
                        f"[CodeNamDungV2] Skipping Simulation for test-derived "
                        f"plan (plan_no={plan_no}).\n"
                    )
            else:
                problem_with_planning = f"## Problem:\n{problem}\n\n{plan}"

                # ── Simulation / Plan Validation ───────────────────────────────
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
                    print(f"[CodeNamDungV2] Input for Simulation: {plan_no}\n")
                    print(input_for_simulation[0]["content"], flush=True)

                response = self.gpt_chat(processed_input=input_for_simulation)

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(f"[CodeNamDungV2] Response from Simulation: {plan_no}\n")
                    print(response, flush=True)

                if (
                    "Plan Modification Needed" in response
                    and "No Plan Modification Needed" not in response
                ):
                    if self.verbose >= VERBOSE_FULL:
                        print("\n\n" + "_" * 70)
                        print("[CodeNamDungV2] Plan Modification Needed.\n")

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
                        print(
                            f"[CodeNamDungV2] Input for Plan Refinement: {plan_no}\n"
                        )
                        print(input_for_plan_refinement[0]["content"], flush=True)

                    plan = self.gpt_chat(processed_input=input_for_plan_refinement)

                    if self.verbose >= VERBOSE_FULL:
                        print("\n\n" + "_" * 70)
                        print(
                            f"[CodeNamDungV2] Response from Plan Refinement: {plan_no}\n"
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
                print("[CodeNamDungV2] Input for Code Generation:\n")
                print(input_for_code_gen[0]["content"], flush=True)

            response = self.gpt_chat(processed_input=input_for_code_gen)

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print("[CodeNamDungV2] Response from Code Generation:\n")
                print(response, flush=True)

            code = parse_response(response)
            passed, test_log = self.check(data_row, additional_io, code)

            if passed:
                break

            # ── Standard Debug loop ────────────────────────────────────────────
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
                        f"[CodeNamDungV2] Input for Standard Debug: "
                        f"{plan_no}, {debug_no}\n"
                    )
                    print(input_for_debugging[0]["content"], flush=True)

                response = self.gpt_chat(processed_input=input_for_debugging)

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(
                        f"[CodeNamDungV2] Response from Standard Debug: "
                        f"{plan_no}, {debug_no}\n"
                    )
                    print(response, flush=True)

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

            # ── Test-Driven Re-planning ────────────────────────────────────────
            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "=" * 70)
                print(
                    f"[CodeNamDungV2] Standard debug exhausted after "
                    f"{self.max_debug_try} attempts. "
                    f"Running Test-Driven Re-planning."
                )
                print("=" * 70, flush=True)

            replan_succeeded = False

            for replan_no in range(1, self.max_replan_try + 1):
                input_for_replan = [
                    {
                        "role": "user",
                        "content": prompt_for_test_driven_replanning.format(
                            num_debug=len(debug_history),
                            test_log=test_log,
                            std_input_prompt=std_input_prompt,
                        ),
                    }
                ]
                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(
                        f"[CodeNamDungV2] Input for Test-Driven Re-planning: "
                        f"{plan_no}, {replan_no}\n"
                    )
                    print(input_for_replan[0]["content"], flush=True)

                replan_response = self.gpt_chat(
                    processed_input=input_for_replan
                )

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(
                        f"[CodeNamDungV2] Response from Test-Driven Re-planning: "
                        f"{plan_no}, {replan_no}\n"
                    )
                    print(replan_response, flush=True)

                new_plan = _extract_new_plan(replan_response)
                if new_plan:
                    test_derived_plan = new_plan
                    replan_succeeded = True
                    if self.verbose >= VERBOSE_FULL:
                        print(
                            f"\n[CodeNamDungV2] Test-derived plan extracted on "
                            f"re-planning {replan_no}. "
                            f"Will be used in the next planning iteration.",
                            flush=True,
                        )
                    break
                else:
                    if self.verbose >= VERBOSE_FULL:
                        print(
                            f"\n[CodeNamDungV2] Re-planning {replan_no}: "
                            f"could not extract new plan. "
                            f"Falling back to enhanced debug.",
                            flush=True,
                        )

            if replan_succeeded:
                # Skip enhanced debug; test_derived_plan carries into next plan_no
                continue

            # ── Enhanced Fault-Localisation Debug (from CodeNamDung) ──────────
            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "=" * 70)
                print(
                    f"[CodeNamDungV2] Switching to Enhanced Fault-Localisation Debug."
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
                        f"[CodeNamDungV2] Input for Enhanced Debug: "
                        f"{plan_no}, {enh_no}\n"
                    )
                    print(input_for_enhanced[0]["content"], flush=True)

                response = self.gpt_chat(processed_input=input_for_enhanced)

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n" + "_" * 70)
                    print(
                        f"[CodeNamDungV2] Response from Enhanced Debug: "
                        f"{plan_no}, {enh_no}\n"
                    )
                    print(response, flush=True)

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
                            f"\n[CodeNamDungV2] Enhanced Debug fixed the bug "
                            f"on attempt {enh_no}!",
                            flush=True,
                        )
                    break

            if passed:
                break

        if self.verbose >= VERBOSE_FULL:
            print("\n\n" + "_" * 70)

        return code
