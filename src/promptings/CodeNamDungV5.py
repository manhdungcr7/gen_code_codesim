"""
CodeNamDungV5 — CodeSIM + Self-Consistency & Consensus Voting

Motivation (from the LBPP post-mortem):
  - 32.7% of failures are SILENT: CodeSIM passes the 2 sample tests
    (LBPP uses only test_list[:2] during debug) and declares success, but the
    code is wrong on hidden inputs.
  - The single LLM run is unstable: the same problem sometimes yields correct
    code, sometimes not. CodeSIM only ever samples ONCE.

Idea (CodeT-style, adapted to be regression-safe):
  1. Generate k INDEPENDENT candidate solutions (needs temperature > 0).
  2. Generate m ranking test cases (assert statements).
  3. Run every candidate against every ranking test → a pass/fail vector.
  4. Consensus: candidates that share the same pass/fail vector "agree".
     score(c) = |agreement group of c| × (#ranking tests c passes)
  5. Pick the highest-scoring candidate that also passes the 2 sample tests.

Why this avoids the V3 regression trap:
  - Ranking tests are used ONLY to SCORE/RANK candidates, never as hard
    constraints the code must satisfy. A wrong generated test fails all
    candidates roughly equally → adds noise, never forces a correct candidate
    to "fix" itself into a wrong one.
  - The 2 real sample tests remain the primary gate; consensus only breaks
    ties among candidates that already pass them — exactly the silent-failure
    scenario.

Requirement: run with --temperature > 0 (e.g. 0.8). At temperature 0 the k
candidates collapse to one and the method degrades gracefully to CodeSIM.
"""

from collections import defaultdict
from typing import List, Tuple

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


# ─── Ranking Test Generation Prompt ─────────────────────────────────────────────

prompt_for_ranking_tests = """\
You are a tester creating unit tests to DISTINGUISH correct from incorrect \
implementations of the problem below.

## Problem
{problem}

**Your task:**
Generate {num_tests} test cases that exercise DIFFERENT and DISCRIMINATING \
scenarios — the kind of inputs where a subtly-wrong implementation would \
behave differently from a correct one:
  - edge cases (empty / single element / minimal input)
  - boundary values
  - cases whose answer is negative / empty / zero
  - larger or structurally varied inputs

For each test case:
  1. Construct any objects the signature needs (e.g. build the tree/list nodes).
  2. Call the function.
  3. Assert the expected result you computed by hand.

Format each test as runnable Python, one assertion per block:
```python
# Test: short description
<setup if needed>
assert <function_call> == <expected_value>
```

--------
Important:
- Do NOT write the solution — only tests.
- One `assert` per test block.
- Generate exactly {num_tests} tests.
"""


# ─── CodeNamDungV5 ───────────────────────────────────────────────────────────────

class CodeNamDungV5(CodeSIM):
    """
    CodeSIM + Self-Consistency & Consensus Voting.

    Parameters
    ----------
    num_candidates : int
        Number of independent candidate solutions to generate.  Default = 3.
    num_ranking_tests : int
        Number of ranking test cases generated for consensus scoring.  Default = 6.
    candidate_debug_try : int
        Debug attempts (against the 2 sample tests) per candidate.  Default = 3.
    All other parameters forwarded to CodeSIM unchanged.

    NOTE: Requires temperature > 0 for the candidates to be diverse.
    """

    def __init__(
        self,
        num_candidates: int = 3,
        num_ranking_tests: int = 6,
        candidate_debug_try: int = 3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_candidates = num_candidates
        self.num_ranking_tests = num_ranking_tests
        self.candidate_debug_try = candidate_debug_try

        if self.verbose >= VERBOSE_FULL:
            print(
                f"[CodeNamDungV5] num_candidates={self.num_candidates}, "
                f"num_ranking_tests={self.num_ranking_tests}, "
                f"candidate_debug_try={self.candidate_debug_try}",
                flush=True,
            )

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _get_setup(self, data_row: dict) -> str:
        """Best-effort retrieval of the test setup (class defs etc.)."""
        decoder = getattr(self.data, "_decode_item", None)
        if decoder is not None:
            try:
                setup, _ = decoder(data_row)
                return setup if isinstance(setup, str) else ""
            except Exception:
                return ""
        return ""

    def _pass_vector(
        self, data_row: dict, code: str, tests: List[str]
    ) -> Tuple[int, ...]:
        """Return a per-test pass(1)/fail(0) vector for `code` on `tests`."""
        vec = []
        for t in tests:
            try:
                passed, _ = self.data.evaluate_additional_io(
                    data_row[self.data.id_key], [t], code, self.language
                )
            except Exception:
                passed = False
            vec.append(1 if passed else 0)
        return tuple(vec)

    def _generate_candidate(
        self,
        data_row: dict,
        problem: str,
        std_input_prompt: str,
        additional_io: List[str],
        idx: int,
    ) -> Tuple[str, bool]:
        """Run one CodeSIM-lite pipeline; return (code, passed_sample_tests)."""

        # ── Planning ───────────────────────────────────────────────────────────
        input_for_planning = [
            {
                "role": "user",
                "content": prompt_for_planning.format(
                    problem=problem, language=self.language
                ),
            }
        ]
        if self.verbose >= VERBOSE_FULL:
            print("\n\n" + "_" * 70)
            print(f"[CodeNamDungV5] Candidate {idx}: Input for Planning\n")
            print(input_for_planning[0]["content"], flush=True)

        response = self.gpt_chat(processed_input=input_for_planning)

        if self.verbose >= VERBOSE_FULL:
            print("\n\n" + "_" * 70)
            print(f"[CodeNamDungV5] Candidate {idx}: Response from Planning\n")
            print(response, flush=True)

        plan = (
            response[response.rfind("### Plan"):]
            if "### Plan" in response
            else f"### Plan\n\n{response}"
        )
        problem_with_planning = f"## Problem:\n{problem}\n\n{plan}"

        # ── Simulation ───────────────────────────────────────────────────────────
        input_for_simulation = [
            {
                "role": "user",
                "content": prompt_for_simulation.format(
                    problem_with_planning=problem_with_planning,
                    language=self.language,
                ),
            }
        ]
        response = self.gpt_chat(processed_input=input_for_simulation)

        if self.verbose >= VERBOSE_FULL:
            print("\n\n" + "_" * 70)
            print(f"[CodeNamDungV5] Candidate {idx}: Response from Simulation\n")
            print(response, flush=True)

        if (
            "Plan Modification Needed" in response
            and "No Plan Modification Needed" not in response
        ):
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
            plan = self.gpt_chat(processed_input=input_for_refinement)
            problem_with_planning = f"## Problem:\n{problem}\n\n{plan}"

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(
                    f"[CodeNamDungV5] Candidate {idx}: Refined plan\n"
                )
                print(plan, flush=True)

        # ── Code Generation ──────────────────────────────────────────────────────
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
        response = self.gpt_chat(processed_input=input_for_code_gen)

        if self.verbose >= VERBOSE_FULL:
            print("\n\n" + "_" * 70)
            print(f"[CodeNamDungV5] Candidate {idx}: Generated code\n")
            print(response, flush=True)

        code = parse_response(response)
        passed, test_log = self.check(data_row, additional_io, code)

        # ── Debug against sample tests ─────────────────────────────────────────
        debug_no = 0
        while not passed and debug_no < self.candidate_debug_try:
            debug_no += 1
            input_for_debug = [
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
            response = self.gpt_chat(processed_input=input_for_debug)

            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "_" * 70)
                print(
                    f"[CodeNamDungV5] Candidate {idx}: Debug {debug_no}\n"
                )
                print(response, flush=True)

            code = parse_response(response)
            passed, test_log = self.check(data_row, additional_io, code)

        return code, passed

    def _select_by_consensus(
        self,
        data_row: dict,
        candidates: List[str],
        sample_passed: List[bool],
        ranking_tests: List[str],
    ) -> str:
        """CodeT-style consensus voting. Returns the winning candidate code."""

        # Per-candidate pass vectors over ranking tests
        vectors = [
            self._pass_vector(data_row, code, ranking_tests)
            for code in candidates
        ]

        # Group candidate indices by identical pass vector (agreement clusters)
        groups: dict = defaultdict(list)
        for i, vec in enumerate(vectors):
            groups[vec].append(i)

        def score(i: int) -> tuple:
            vec = vectors[i]
            group_size = len(groups[vec])
            tests_passed = sum(vec)
            # Prefer: passes sample tests, then larger agreement group,
            # then more ranking tests passed.
            return (int(sample_passed[i]), group_size, tests_passed)

        best_idx = max(range(len(candidates)), key=score)

        if self.verbose >= VERBOSE_FULL:
            print("\n\n" + "=" * 70)
            print("[CodeNamDungV5] Consensus voting results:")
            for i in range(len(candidates)):
                print(
                    f"  Candidate {i}: sample_passed={sample_passed[i]}, "
                    f"pass_vector={vectors[i]}, "
                    f"group_size={len(groups[vectors[i]])}, "
                    f"tests_passed={sum(vectors[i])}, "
                    f"score={score(i)}"
                )
            print(f"  → Winner: Candidate {best_idx}")
            print("=" * 70, flush=True)

        return candidates[best_idx]

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

        # ── Phase 1: Generate k independent candidates ─────────────────────────
        candidates: List[str] = []
        sample_passed: List[bool] = []

        for idx in range(1, self.num_candidates + 1):
            if self.verbose >= VERBOSE_FULL:
                print("\n\n" + "=" * 70)
                print(
                    f"[CodeNamDungV5] Generating candidate "
                    f"{idx}/{self.num_candidates}"
                )
                print("=" * 70, flush=True)

            code, passed = self._generate_candidate(
                data_row, problem, std_input_prompt, additional_io, idx
            )
            candidates.append(code)
            sample_passed.append(passed)

            # Early exit not taken: we want diversity for voting even if one passes.

        # If any candidate already passes sample tests and there is no diversity
        # to exploit, short-circuit to the first passing one.
        unique_codes = set(c.strip() for c in candidates)
        if len(unique_codes) == 1:
            if self.verbose >= VERBOSE_FULL:
                print(
                    "\n[CodeNamDungV5] All candidates identical "
                    "(temperature likely 0). Returning the single solution.",
                    flush=True,
                )
            return candidates[0]

        # ── Phase 2: Generate ranking tests ────────────────────────────────────
        if self.verbose >= VERBOSE_FULL:
            print("\n\n" + "=" * 70)
            print(
                f"[CodeNamDungV5] Generating {self.num_ranking_tests} "
                f"ranking tests for consensus."
            )
            print("=" * 70, flush=True)

        input_for_ranking = [
            {
                "role": "user",
                "content": prompt_for_ranking_tests.format(
                    problem=problem,
                    num_tests=self.num_ranking_tests,
                ),
            }
        ]
        if self.verbose >= VERBOSE_FULL:
            print("\n\n" + "_" * 70)
            print("[CodeNamDungV5] Input for Ranking Test Generation:\n")
            print(input_for_ranking[0]["content"], flush=True)

        ranking_response = self.gpt_chat(processed_input=input_for_ranking)

        if self.verbose >= VERBOSE_FULL:
            print("\n\n" + "_" * 70)
            print("[CodeNamDungV5] Response from Ranking Test Generation:\n")
            print(ranking_response, flush=True)

        parsed_tests = self.parse_test_cases(ranking_response)

        # Prepend setup (class defs etc.) so tests with custom types can run
        setup = self._get_setup(data_row)
        if setup:
            ranking_tests = [f"{setup}\n{t}" for t in parsed_tests]
        else:
            ranking_tests = parsed_tests

        # ── Phase 3: Consensus voting ──────────────────────────────────────────
        if not ranking_tests:
            # No ranking tests parsed → fall back to any sample-passing candidate
            if self.verbose >= VERBOSE_FULL:
                print(
                    "\n[CodeNamDungV5] No ranking tests parsed. "
                    "Falling back to first sample-passing candidate.",
                    flush=True,
                )
            for i, ok in enumerate(sample_passed):
                if ok:
                    return candidates[i]
            return candidates[0]

        best_code = self._select_by_consensus(
            data_row, candidates, sample_passed, ranking_tests
        )

        if self.verbose >= VERBOSE_FULL:
            print("\n\n" + "_" * 70)

        return best_code
