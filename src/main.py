import dotenv
dotenv.load_dotenv(override=True)

import argparse
import sys
from datetime import datetime
from constants.paths import *

from models.Gemini import Gemini
from models.OpenAI import OpenAIModel

from results.Results import Results

from promptings.PromptingFactory import PromptingFactory
from datasets.DatasetFactory import DatasetFactory
from models.ModelFactory import ModelFactory

from constants.verboseType import *

from utils.summary import gen_summary
from utils.runEP import run_eval_plus
from utils.evaluateET import generate_et_dataset_human
from utils.evaluateET import generate_et_dataset_mbpp
from utils.generateEP import generate_ep_dataset_human
from utils.generateEP import generate_ep_dataset_mbpp

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    default="HumanEval",
    choices=[
        "HumanEval",
        "MBPP",
        "APPS",
        "xCodeEval",
        "CC",
        "BigCodeBench-Hard",
        "LBPP"
    ]
)
parser.add_argument(
    "--strategy",
    type=str,
    default="Direct",
    choices=[
        "Direct",
        "CoT",
        "SelfPlanning",
        "Analogical",
        "MapCoder",
        "CodeSIM",
        "CodeNamDung",
        "CodeNamDungV2",
        "CodeNamDungV3",
        "CodeNamDungV4",
        "CodeNamDungV5",
        "CodeSIMWD",
        "CodeSIMWPV",
        "CodeSIMWPVD",
        "CodeSIMA",
        "CodeSIMC",
    ]
)
parser.add_argument(
    "--model",
    type=str,
    default="ChatGPT",
)
parser.add_argument(
    "--model_provider",
    type=str,
    default="OpenAI",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0
)
parser.add_argument(
    "--top_p",
    type=float,
    default=0.95
)
parser.add_argument(
    "--pass_at_k",
    type=int,
    default=1
)
parser.add_argument(
    "--language",
    type=str,
    default="Python3",
    choices=[
        "C",
        "C#",
        "C++",
        "Go",
        "PHP",
        "Python3",
        "Ruby",
        "Rust",
    ]
)

parser.add_argument(
    "--cont",
    type=str,
    default="yes",
    choices=[
        "yes",
        "no"
    ]
)

parser.add_argument(
    "--result_log",
    type=str,
    default="partial",
    choices=[
        "full",
        "partial"
    ]
)

parser.add_argument(
    "--verbose",
    type=str,
    default="2",
    choices=[
        "2",
        "1",
        "0",
    ]
)

parser.add_argument(
    "--store_log_in_file",
    type=str,
    default="yes",
    choices=[
        "yes",
        "no",
    ]
)

args = parser.parse_args()

DATASET = args.dataset
STRATEGY = args.strategy
MODEL_NAME = args.model
MODEL_PROVIDER_NAME = args.model_provider
TEMPERATURE = args.temperature
TOP_P = args.top_p
PASS_AT_K = args.pass_at_k
LANGUAGE = args.language
CONTINUE = args.cont
RESULT_LOG_MODE = args.result_log
VERBOSE = int(args.verbose)
STORE_LOG_IN_FILE = args.store_log_in_file

MODEL_NAME_FOR_RUN = MODEL_NAME

RUN_NAME = f"results/{DATASET}/{STRATEGY}/{MODEL_NAME_FOR_RUN}/{LANGUAGE}-{TEMPERATURE}-{TOP_P}-{PASS_AT_K}"

run_no = 1
while os.path.exists(f"{RUN_NAME}/Run-{run_no}"):
    run_no += 1

if CONTINUE == "yes" and run_no > 1:
    run_no -= 1

RUN_NAME = f"{RUN_NAME}/Run-{run_no}"

if not os.path.exists(RUN_NAME):
    os.makedirs(RUN_NAME)

RESULTS_PATH = f"{RUN_NAME}/Results.jsonl"
SUMMARY_PATH = f"{RUN_NAME}/Summary.txt"
LOGS_PATH = f"{RUN_NAME}/Log.txt"

if STORE_LOG_IN_FILE.lower() == 'yes':
    sys.stdout = open(
        LOGS_PATH,
        mode="a",
        encoding="utf-8"
    )

if CONTINUE == "no" and VERBOSE >= VERBOSE_MINIMAL:
    print(f"""
##################################################
Experiment start {RUN_NAME}, Time: {datetime.now()}
###################################################
""")

strategy = PromptingFactory.get_prompting_class(STRATEGY)(
    model=ModelFactory.get_model_class(MODEL_PROVIDER_NAME)(
        model_name=MODEL_NAME, 
        temperature=TEMPERATURE, 
        top_p=TOP_P
    ),
    data=DatasetFactory.get_dataset_class(DATASET)(),
    language=LANGUAGE,
    pass_at_k=PASS_AT_K,
    results=Results(RESULTS_PATH),
    verbose=VERBOSE
)

strategy.run(RESULT_LOG_MODE.lower() == 'full')

if VERBOSE >= VERBOSE_MINIMAL:
    print(f"""
##################################################
Experiment end {RUN_NAME}, Time: {datetime.now()}
###################################################
""")

gen_summary(RESULTS_PATH, SUMMARY_PATH)

ET_RESULTS_PATH = f"{RUN_NAME}/Results-ET.jsonl"
ET_SUMMARY_PATH = f"{RUN_NAME}/Summary-ET.txt"

EP_RESULTS_PATH = f"{RUN_NAME}/Results-EP.jsonl"
EP_SUMMARY_PATH = f"{RUN_NAME}/Summary-EP.txt"

if "human" in DATASET.lower():
    generate_et_dataset_human(RESULTS_PATH, ET_RESULTS_PATH)
    gen_summary(ET_RESULTS_PATH, ET_SUMMARY_PATH)

    # generate_ep_dataset_human(RESULTS_PATH, EP_RESULTS_PATH)
    # run_eval_plus(EP_RESULTS_PATH, EP_SUMMARY_PATH, "humaneval")

elif "mbpp" in DATASET.lower():
    generate_et_dataset_mbpp(RESULTS_PATH, ET_RESULTS_PATH)
    gen_summary(ET_RESULTS_PATH, ET_SUMMARY_PATH)

    # generate_ep_dataset_human(RESULTS_PATH, EP_RESULTS_PATH)
    # run_eval_plus(EP_RESULTS_PATH, EP_SUMMARY_PATH, "mbpp")

if STORE_LOG_IN_FILE.lower() == 'yes':
    sys.stdout.close()

