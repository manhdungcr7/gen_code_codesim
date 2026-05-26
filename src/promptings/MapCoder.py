from typing import List
import tiktoken
import os
import json
import re
import sys
import time

from copy import deepcopy
import xml.etree.ElementTree as ET

from .Base import BaseStrategy
from models.Base import BaseModel

from datasets.Dataset import Dataset
from datasets.APPSDataset import APPSDataset
from datasets.MBPPDataset import MBPPDataset
from datasets.XCodeDataset import XCodeDataset
from datasets.HumanEvalDataset import HumanDataset
from datasets.CodeContestDataset import CodeContestDataset

from results.Results import Results
from evaluations.func_evaluate import evaluate_io
from constants.verboseType import *

mapping = {
    1: "one (01)",
    2: "two (02)",
    3: "three (03)",
    4: "four (04)",
    5: "five (05)",
    6: "six (06)",
    7: "seven (07)",
    8: "eight (08)",
    9: "nine (09)",
}

# KB + Exemplars + Example Planning + Problem Planning + Code Generation + Sample IO testing + Code Improvement


class MapCoder(BaseStrategy):
    def __init__(
        self,
        k: int = 3,
        t: int = 5,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.t = t

    def xml_to_dict(self, element):
        result = {}
        for child in element:
            if child:
                child_data = self.xml_to_dict(child)
                if child.tag in result:
                    if isinstance(result[child.tag], list):
                        result[child.tag].append(child_data)
                    else:
                        result[child.tag] = [result[child.tag], child_data]
                else:
                    result[child.tag] = child_data
            else:
                result[child.tag] = child.text
        return result

    def parse_xml(self, response: str) -> dict:
        if '```xml' in response:
            response = response.replace('```xml', '')
        if '```' in response:
            response = response.replace('```', '')

        try:
            # Thử parse bằng ElementTree như cũ
            root = ET.fromstring(response)
            return self.xml_to_dict(root)
        except Exception:
            try:
                root = ET.fromstring('<root>\n' + response + '\n</root>')
                return self.xml_to_dict(root)
            except Exception:
                # Nếu ElementTree thất bại do XML lỗi, dùng BeautifulSoup
                try:
                    from bs4 import BeautifulSoup
                    # Khởi tạo BeautifulSoup với nội dung bọc trong <root>
                    soup = BeautifulSoup(f"<root>{response}</root>", "html.parser")
                    
                    def soup_to_dict(element):
                        res = {}
                        for child in element.find_all(recursive=False):
                            # Nếu child CÓ thẻ con bên trong thì đệ quy, nếu không thì lấy chữ
                            child_data = soup_to_dict(child) if child.find_all(recursive=False) else child.get_text(strip=True)
                            
                            if child.name:
                                tag_name = child.name
                                if tag_name in res:
                                    if isinstance(res[tag_name], list):
                                        res[tag_name].append(child_data)
                                    else:
                                        res[tag_name] = [res[tag_name], child_data]
                                else:
                                    res[tag_name] = child_data
                        return res
                    
                    root_node = soup.find('root')
                    if root_node:
                        return soup_to_dict(root_node)
                    return soup_to_dict(soup)
                except Exception as e:
                    print(f"\n[Cảnh báo] Không thể parse XML từ response: {e}")
                    return {}

    def parse_code(self, response: str) -> str:
        if "```" not in response:
            return response

        code_pattern = r'```((.|\n)*?)```'
        if "```Python" in response:
            code_pattern = r'```Python((.|\n)*?)```'
        if "```Python3" in response:
            code_pattern = r'```Python3((.|\n)*?)```'
        if "```python" in response:
            code_pattern = r'```python((.|\n)*?)```'
        if "```python3" in response:
            code_pattern = r'```python3((.|\n)*?)```'
        if "```C" in response:
            code_pattern = r'```C((.|\n)*?)```'
        if "```c" in response:
            code_pattern = r'```c((.|\n)*?)```'
        if "```C++" in response:
            code_pattern = r'```C\+\+((.|\n)*?)```'
        if "```c++" in response:
            code_pattern = r'```c\+\+((.|\n)*?)```'
        if "```Java" in response:
            code_pattern = r'```Java((.|\n)*?)```'
        if "```java" in response:
            code_pattern = r'```java((.|\n)*?)```'
        if "```Node" in response:
            code_pattern = r'```Node((.|\n)*?)```'
        if "```node" in response:
            code_pattern = r'```node((.|\n)*?)```'
        if "```Rust" in response:
            code_pattern = r'```Rust((.|\n)*?)```'
        if "```rust" in response:
            code_pattern = r'```rust((.|\n)*?)```'
        if "```PHP" in response:
            code_pattern = r'```PHP((.|\n)*?)```'
        if "```php" in response:
            code_pattern = r'```php((.|\n)*?)```'
        if "```Go" in response:
            code_pattern = r'```Go((.|\n)*?)```'
        if "```go" in response:
            code_pattern = r'```go((.|\n)*?)```'
        if "```Ruby" in response:
            code_pattern = r'```Ruby((.|\n)*?)```'
        if "```ruby" in response:
            code_pattern = r'```ruby((.|\n)*?)```'
        if "```C#" in response:
            code_pattern = r'```C#((.|\n)*?)```'
        if "```c#" in response:
            code_pattern = r'```c#((.|\n)*?)```'
        if "```csharp" in response:
            code_pattern = r'```csharp((.|\n)*?)```'

        code_blocks = re.findall(code_pattern, response, re.DOTALL)

        if type(code_blocks[-1]) == tuple or type(code_blocks[-1]) == list:
            code_str = "\n".join(code_blocks[-1])
        elif type(code_blocks[-1]) == str:
            code_str = code_blocks[-1]
        else:
            code_str = response

        return code_str

    @staticmethod
    def trim_text(text: str, trimmed_text: str):
        return text.replace(trimmed_text, '').strip()

    @staticmethod
    def replace_tag(text: str, tag: str):
        if f'<{tag}><![CDATA[' in text and f']]></{tag}>' in text:
            return text 
        else:
            return text.replace(f'<{tag}>', f'<{tag}><![CDATA[').replace(f'</{tag}>', f']]></{tag}>').strip()

    @staticmethod
    def get_sample_io_str(sample_io: any) -> str:
        if len(sample_io) > 0:
            if type(sample_io[0]) == str:
                return "\n".join(sample_io)
            if type(sample_io[0]) == dict:
                return "\n".join([f"Input:\n{io['input']}\nExpected output:\n{io['output'][0]}" for io in sample_io])
        return sample_io

    def run_single_pass(self, data_row: dict):
        if self.verbose >= VERBOSE_FULL:
            print("", flush=True)

        input_kb_exemplars = [
            {
                "role": "user",
                "content": f"""Given a problem, provide relevant problems then identify the algorithm behind it and also explain the tutorial of the algorithm.
# Problem:
{self.data.get_prompt(data_row)}

# Exemplars:
Recall {mapping[self.k]} relevant and distinct problems (different from problem mentioned above). For each problem,
1. describe it
2. generate {self.language} code step by step to solve that problem
3. finally generate a planning to solve that problem

# Algorithm:

----------------
Important:
Your response must follow the following xml format-

<root>
<problem>
# Recall {mapping[self.k]} relevant and distinct problems (different from problem mentioned above). Write each problem in the following format.
<description>
# Describe the problem.
</description>
<code>
# Let's think step by step to solve this problem in {self.language} programming language.
</code>
<planning>
# Planning to solve this problem.
</planning>
</problem>

# similarly add more problems here...

<algorithm>
# Identify the algorithm (Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, and so on) that needs to be used to solve the original problem.
# Write a useful tutorial about the above mentioned algorithms. Provide a high level generic tutorial for solving this types of problem. Do not generate code.
</algorithm>
</root>
""",
            },
        ]

        if self.verbose >= VERBOSE_FULL:
            print("\n\n________________________")
            print("Input for knowledge base and exemplars: ")
            print(input_kb_exemplars[0]['content'], flush=True)

        response = self.gpt_chat(
            processed_input=input_kb_exemplars
        )

        # Post processing
        response = self.trim_text(
            response, "# Identify the algorithm (Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, Recursive, Binary search, and so on) that needs to be used to solve the original problem.")
        response = self.trim_text(
            response, "# Write a useful tutorial about the above mentioned algorithms. Provide a high level generic tutorial for solving this types of problem. Do not generate code.")
        response = self.trim_text(
            response, "# Planning to solve this problem:")
        response = self.trim_text(
            response, f"# Let's think step by step to solve this problem in {self.language} programming language.")
        response = self.replace_tag(response, 'algorithm')
        response = self.replace_tag(response, 'description')
        response = self.replace_tag(response, 'code')
        response = self.replace_tag(response, 'planning')

        if self.verbose >= VERBOSE_FULL:
            print("\n\n________________________")
            print("Response from knowledge base and exemplars: ")
            print(response, flush=True)

        response = self.parse_xml(response)

        # Safeguard if the model fails to return the exact tags properly
        algo = response.get('algorithm', '')
        algorithm_prompt = f"## Relevant Algorithm to solve the next problem:\n{algo}" if algo else ""
        
        # Also safeguard 'problem' key to prevent similar KeyErrors in the next loop
        problems_list = response.get('problem', [])
        if not isinstance(problems_list, list):
            # Sometimes LLM might generate a single problem dict rather than a list
            problems_list = [problems_list] if problems_list else []
        response['problem'] = problems_list
        sample_io = data_row.get('sample_io', [])
        sample_io_str = self.get_sample_io_str(sample_io)
        if sample_io_str and sample_io_str != "[]" and sample_io_str != []:
            sample_io_prompt = f"## Sample Test cases: \n{sample_io_str}\n" 
        else:
            sample_io_prompt = ""

        # if type(self.data) != MBPPDataset and type(self.data) != XCodeDataset else ""

        plannings = []
        for example_no, example in enumerate(response["problem"], start=1):
            example_problem = example["description"]
            example_planning = example["planning"]

            input_for_problem_planning = [
                {
                    "role": "user",
                    "content": f"Given a competitive programming problem generate a concrete planning to solve the problem.\n# Problem:\n{example_problem}\n# Planning:\n{example_planning}\n{algorithm_prompt}\n## Problem to be solved:\n{self.data.get_prompt(data_row)}\n{sample_io_prompt}\n## Planning:\n\n----------------\nImportant: You should give only the planning to solve the problem. Do not add extra explanation or words."
                }
            ]

            if self.verbose >= VERBOSE_FULL:
                print("\n\n________________________")
                print(
                    f"Input for our problem planning using example: {example_no}: ")
                print(input_for_problem_planning[0]['content'], flush=True)

            planning = self.gpt_chat(
                input_for_problem_planning
            )

            # planning = self.parse_xml(planning)
            # planning['confidence'] = int(str(planning['confidence']).strip())

            if self.verbose >= VERBOSE_FULL:
                print("\n\n________________________")
                print("Response from our problem planning: ")
                print(planning, flush=True)

            input_for_planning_verification = [
                {
                    "role": "user",
                    "content": f"Given a competitive programming problem and a plan to solve the problem in {self.language}, tell whether the plan is correct to solve this problem.\n\n# Problem:\n{self.data.get_prompt(data_row)}\n# Planning:\n{planning}\n\n----------------\nImportant: Your response must follow the following xml format-```\n<root>\n<explanation> Discuss whether the given competitive programming problem is solvable by using the above mentioned planning.</explanation>\n<confidence> Confidence score regarding the solvability of the problem. Must be an integer between 0 and 100. </confidence>\n</root>\n```"
                }
            ]

            if self.verbose >= VERBOSE_FULL:
                print("Input for planning verification: ")
                print(input_for_planning_verification[0]['content'], flush=True)

            verification_res = self.gpt_chat(
                input_for_planning_verification
            )

            verification_res = self.replace_tag(
                verification_res, 'explanation')
            verification_res = self.replace_tag(verification_res, 'confidence')

            verification_res = self.parse_xml(verification_res)

            verification_res['confidence'] = int(
                str(verification_res['confidence']).strip())

            if self.verbose >= VERBOSE_FULL:
                print("Response from planning verification: ")
                print(verification_res, flush=True)

            plannings.append((
                planning,
                verification_res['confidence'],
                example
            ))

            # if type(self.data) == MBPPDataset and verification_res['confidence'] == 100:
            #     break

        plannings.sort(key=lambda x: x[1], reverse=True)
        # time.sleep(1)

        if type(self.data) == APPSDataset or type(self.data) == CodeContestDataset or type(self.data) == XCodeDataset:
            std_input_prompt = "## Note: Strictly follow the input and output format. The input should be taken from Standard input and output should be given to standard output. If you are writing a function then after the function definition take input using `input()` function then call the function with specified parameters and finally print the output of the function. Do not add extra print statement otherwise it will failed the test cases."
        else:
            std_input_prompt = ""

        for planning_with_ex in plannings:
            planning, confidence, example = planning_with_ex

            input_for_final_code_generation = [
                {
                    "role": "user",
                    "content": f"Given a competitive programming problem generate {self.language} code to solve the problem.\n{algorithm_prompt}\n## Problem to be solved:\n{self.data.get_prompt(data_row)}\n## Planning:\n{planning}\n{sample_io_prompt}\n## Let's think step by step.\n\n----------------\nImportant:\n{std_input_prompt}\n## Your response must contain only the {self.language} code to solve this problem. Do not add extra explanation or words."
                }
            ]

            if self.verbose >= VERBOSE_FULL:
                print("\n\n________________________")
                print("Input for final code generation: ")
                print(input_for_final_code_generation[0]['content'], flush=True)

            code = self.gpt_chat(
                input_for_final_code_generation
            )

            code = self.parse_code(code)

            if self.verbose >= VERBOSE_FULL:
                print("\n\n________________________")
                print("Response from final code generation: ")
                print(code, flush=True)

            response = f"## Planning: {planning}\n## Code:\n```\n{code}\n```"
            passed = False

            for i in range(1, self.t + 1):
                passed, test_log = self.data.evaluate_sample_io(
                    data_row,
                    code,
                    self.language
                )

                if passed:
                    break

                if self.verbose >= VERBOSE_FULL:
                    print(f"Input for improving code generation: {i}")
                input_for_improving_code = [
                    {
                        "role": "user",
                        "content": f"Given a competitive programming problem you have generated {self.language} code to solve the problem. But the generated code can not pass sample test cases. Improve your code to solve the problem correctly.\n{algorithm_prompt}\n## Problem to be solved:\n{self.data.get_prompt(data_row)}\n{response}\n## Test Report:\n{test_log}\n## Modified Planning:\n## Let's think step by step to modify {self.language} Code for solving this problem.\n\n----------------\nImportant:\n{std_input_prompt}\n## Your response must contain the modified planning and then the {self.language} code inside ``` block to solve this problem."
                    }
                ]

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n________________________")
                    print("Input for improving code generation: ")
                    print(input_for_improving_code[0]['content'], flush=True)

                response = self.gpt_chat(
                    input_for_improving_code
                )
                code = self.parse_code(response)

                if self.verbose >= VERBOSE_FULL:
                    print("\n\n________________________")
                    print("Response from improving code generation: ")
                    print(response, flush=True)

            # got a code that passed all sample test cases
            if passed:
                break

        if self.verbose >= VERBOSE_FULL:
            print("________________________\n\n", flush=True)
        
        return code
