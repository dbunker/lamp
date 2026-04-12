from dataclasses import dataclass
from typing import List, Set, Tuple, Dict, Optional, Iterable
import argparse
import random
import os
import json
import re
import time
import subprocess
import clingo
from abc import ABC, abstractmethod
from openai import OpenAI
from pathlib import Path
from ollama import Client
import pandas as pd
import logging
from math import comb

DATA_FOLDER = "data"
TIMEOUT = 420
SEARCH_SPACE_DEMO_LIMIT = 30000

SPACE_TIMEOUT = 30

########## ASP Models ##########

@dataclass(frozen=True)
class Atom:
    pred: str
    terms: Tuple[str, ...]

    def is_ground(self) -> bool:
        return all(arg[0].islower() for arg in self.terms)

    def __str__(self):
        terms_str = ", ".join(str(arg) for arg in self.terms)
        return f"{self.pred}({terms_str})"

@dataclass(frozen=True)
class Literal:
    pos: bool
    atom: Atom

    def __str__(self):
        return f"{'not ' if not self.pos else ''}{self.atom}"

@dataclass(frozen=True)
class Rule:
    head: Atom
    body: Set[Literal]

    def __str__(self):
        if len(self.body) == 0:
            return f"{self.head}."

        body_str = ", ".join(str(atom) for atom in self.body)
        return f"{self.head} :- {body_str}."

@dataclass
class KnowledgeBase:
    facts: List[Atom]
    rules: List[Rule]

    def __str__(self):
        facts = [f"{str(fact)}." for fact in self.facts]
        rules = [str(rule) for rule in self.rules]
        return "\n".join(facts + [""] + rules)


atom_re = re.compile(r"\s*([A-Za-z_]\w*)\s*\(\s*(.*?)\s*\)\s*$")

def parse_atom(atom_string: str) -> Atom:

    match = atom_re.fullmatch(atom_string)
    if not match:
        raise ValueError(f"Not an Atom: {atom_string!r}")

    pred, terms_str = match.groups()
    terms = tuple(a.strip() for a in terms_str.split(",")) if terms_str else ()

    if len(terms) == 1 and terms[0] == "":
        terms = ()

    return Atom(pred, terms)


def split_body_literals(body_str: str) -> list:
    """Split body literals by commas, ignoring commas inside parentheses."""
    parts = []
    depth = 0
    current = []
    for ch in body_str:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current))
    return parts


def parse_rule(rule_string: str) -> Rule:
    rule_string = rule_string.strip().rstrip(".")

    if ":-" in rule_string:
        head_str, body_str = map(str.strip, rule_string.split(":-", 1))
        lits = []

        for part in split_body_literals(body_str):

            part = part.strip()
            pos = not part.startswith("not ")
            if not pos:
                part = part[4:].strip()
            lits.append(Literal(pos, parse_atom(part)))

        return Rule(parse_atom(head_str), set(lits))

    else:
        return Rule(parse_atom(rule_string), set())


def parse_kb(program: str) -> KnowledgeBase:
    # split by "." that terminate statements

    stmts = [t.strip() for t in program.split(".") if t.strip()]
    facts: List[Atom] = []
    rules: List[Rule] = []
    for stmt in stmts:
        if ":-" in stmt:
            rules.append(parse_rule(stmt))
        else:
            facts.append(parse_atom(stmt))
    return KnowledgeBase(facts=facts, rules=rules)

########## LLM ##########

class LLM(ABC):

    @abstractmethod
    def send_prompt(self):
        pass


    @abstractmethod
    def get_content(self, full_response: str):
        pass

class OpenAIClient(LLM):
    client: OpenAI
    title: str

    def __init__(self, title):
        self.title = title
        key_obj = json.loads(read_file("key.json"))
        key = key_obj["openai"]
        self.client = OpenAI(api_key=key, timeout=TIMEOUT)

    def send_prompt(self, messages: list) -> str:

        # frequency_penalty=1.0,
        # presence_penalty=1.0,
        # temperature=1.0,
        # reasoning={"effort": "high"},
        # max_output_tokens=1000

        response = self.client.responses.create(
            model=self.title,
            input=messages
        )

        full_response = json.loads(response.model_dump_json())
        logging.info(json.dumps(full_response, indent=2))
        msg_item = next(item for item in full_response["output"] if item.get("type") == "message")
        response_content = msg_item["content"][0]["text"]

        return [full_response, response_content]


    def get_content(self, full_response: str):
        msg_item = next(item for item in full_response["output"] if item.get("type") == "message")
        return msg_item["content"][0]["text"]

class OllamaClient(LLM):
    client: Client
    title: str

    def __init__(self, title):
        self.client = Client(timeout=TIMEOUT)
        self.title = title

    # think="high"

    def send_prompt(self, messages: list) -> str:

        response = self.client.chat(
            model=self.title,
            messages=messages,
        )

        full_response = json.loads(response.model_dump_json())
        response_content = full_response["message"]["content"]

        return [full_response, response_content]


    def get_content(self, full_response):

        return full_response["message"]["content"]


def get_llm_response(client: LLM, messages: list, response_path: str, rerun: bool):

    logging.info(f"Run {client.title}")

    # Check if output is already present, do not run if it is
    if not os.path.exists(response_path) or rerun:

        try:
            _llm_call_start = time.time()
            [full_response, response_content] = client.send_prompt(messages)
            full_response["wall_time_seconds"] = time.time() - _llm_call_start
        except Exception as e:
            logging.warning(f"LLM call timed out or failed: {e}")
            return None

        logging.info(full_response)
        write_json(response_path, full_response)

    else:
        full_response = read_json(response_path)
        response_content = client.get_content(full_response)

    logging.info("Response:")
    logging.info(response_content)
    return response_content

########## Example Generation ##########

@dataclass
class ModelConfig:

    example_number: int

    num_predicates: int
    num_possible_terms: int

    num_facts: int
    num_rules: int

    num_literals: int
    num_neg_literals: int

    min_derived_atoms: int = 1

# Generator creates instances and rules
# d0(o0).
# d0(o1).
# d1(o1).
# d2(X) :- d0(X), not d1(X).

# Example base
# vehicle(scooter).
# vehicle(car).
# slow_moving(scooter).
# highway_possible(X) :- vehicle(X), not slow_moving(X).
def generate_example(config: ModelConfig) -> KnowledgeBase:

    # All possible atoms
    possible_atoms = set()
    predicates = set()

    for desc_id in range(config.num_predicates):
        predicates.add(f"d{desc_id}")

        for obj_id in range(config.num_possible_terms):
            possible_atoms.add(Atom(f"d{desc_id}", (f"o{obj_id}", )))

    # Generate instance
    facts = set(random.sample(list(possible_atoms), config.num_facts))

    # Generate rules
    rules = []
    for rule_id in range(config.num_rules):

        head_predicate = random.sample(list(predicates), 1)[0]
        head = Atom(head_predicate, ("X",))
        remaining_pred = list(predicates - set([head_predicate]))

        body = set()
        for literal_id in range(config.num_literals):

            # Select an unused atom
            to_ground_pred = random.sample(remaining_pred, 1)[0]
            atom = Atom(to_ground_pred, ("X",))

            # Ensure at least one positive and 
            if config.num_literals <= config.num_neg_literals:
                raise RuntimeError("Too many negative literals")

            # Possibly allow for negative
            pos = literal_id <= config.num_literals - config.num_neg_literals - 1
            body.add(Literal(pos, atom))
            remaining_pred = list(set(remaining_pred) - set([to_ground_pred]))

        rule = Rule(head, body)
        rules.append(rule)

    return KnowledgeBase(list(facts), rules)


def generate_benchmark(config: ModelConfig, index):

    kb = generate_example(config)
    model_stats = run_asp(str(kb))

    # Check for no more or less than one stable model
    def derived_atoms(stats):
        return len(stats["witnesses"][0]["atoms"]) - len(kb.facts)

    while (len(model_stats["witnesses"]) != 1 or
           len(model_stats["warnings"]) != 0 or
           derived_atoms(model_stats) < config.min_derived_atoms):

        if len(model_stats["witnesses"]) != 1:
            logging.info(f"Not only 1 stable model: {model_stats['witnesses']}")

        elif len(model_stats["warnings"]) != 0:
            logging.info(f"Has warnings: {model_stats['warnings']}")

        elif derived_atoms(model_stats) < config.min_derived_atoms:
            logging.info(f"Too few derived atoms ({derived_atoms(model_stats)} < {config.min_derived_atoms}): {model_stats['witnesses'][0]['atoms']}")

        kb = generate_example(config)
        model_stats = run_asp(str(kb))

    benchmark_path = f"{DATA_FOLDER}/orig_benchmarks/benchmark_{index}.lp"
    solution_path = f"{DATA_FOLDER}/orig_solutions/solution_{index}.json"

    write_file(benchmark_path, str(kb))
    write_json(solution_path, model_stats)


def generate_benchmarks():

    # Generate rules based on various configurations
    predicates_range     = [10, 6]
    possible_terms_range = [10, 6]
    facts_range          = [9, 6, 5]
    rules_range          = [11]
    literals_range       = [2, 3]
    neg_literals_range   = [0, 1, 2]
    min_derived_atoms_range = [1, 3]

    index = 0
    for example_number in range(1):
        for num_predicates in predicates_range:
            for num_possible_terms in possible_terms_range:
                for num_facts in facts_range:
                    for num_rules in rules_range:
                        for num_literals in literals_range:
                            for num_neg_literals in neg_literals_range:
                                for min_derived_atoms in min_derived_atoms_range:

                                    if num_literals <= num_neg_literals:
                                        continue

                                    config = ModelConfig(
                                        example_number,
                                        num_predicates,
                                        num_possible_terms,
                                        num_facts,
                                        num_rules,
                                        num_literals,
                                        num_neg_literals,
                                        min_derived_atoms
                                    )

                                    generate_benchmark(config, index)
                                    index += 1

########## Utilities ##########

def write_file(path_str: str, text: str):
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_json(path_str: str, data: Dict):
    write_file(path_str, json.dumps(data, indent=2))


def read_file(path_str: str) -> str:
    return Path(path_str).read_text()


def read_json(path_str: str) -> Dict:
    return json.loads(read_file(path_str))

########## Run Clingo ##########

# Run through clingo to get stable model
# d0(o0) d0(o1) d1(o1) d2(o0)
def run_asp(program: str) -> Dict:

    warnings = []
    witnesses = []

    def logger(code, message):
        warnings.append({"code": code.name, "message": message})

    ctl = clingo.Control(["-n", "0", "--warn=all"], logger=logger)

    try:
        ctl.add("base", [], program)
    except RuntimeError as err:
        return {
            "errors": str(err),
            "witnesses": []
        }

    try:
        ctl.ground([("base", [])])
    except RuntimeError as err:
        return {
            "errors": str(err),
            "witnesses": []
        }

    start_time = time.time()

    def on_model(m: clingo.Model):
        t = time.time() - start_time
        atoms = [str(a) for a in m.symbols(shown=True)]
        witnesses.append({
            "index": len(witnesses) + 1,
            "atoms": atoms,
            "time_seconds": round(t, 6)
        })

    result = ctl.solve(on_model=on_model)
    total_runtime = time.time() - start_time

    # Extract statistics dictionary from clingo
    stats = ctl.statistics  
    solver_stats = {
        "models_found": len(witnesses),
        "conflicts": stats.get("solving", {}).get("solvers", {}).get("conflicts", None),
        "choices": stats.get("solving", {}).get("solvers", {}).get("choices", None),
        "atoms": stats.get("problem", {}).get("lp", {}).get("atoms", None),
        "rules": stats.get("problem", {}).get("lp", {}).get("rules", None),
        "time_total": round(stats.get("summary", {}).get("times", {}).get("total", total_runtime), 6),
        "time_solve": round(stats.get("summary", {}).get("times", {}).get("solve", 0), 6),
        "time_total_python": round(total_runtime, 0),
    }

    model_stats = {
        "satisfiable": result.satisfiable,
        "witnesses": witnesses,
        "warnings": warnings,
        "statistics": solver_stats
    }

    return model_stats


# Get stable models from clingo output json
def expected_models(json_response: Dict) -> List[Atom]:
    models = json_response["Call"][0]["Witnesses"]

    if len(models) != 1:
        raise RuntimeError(f"More or less than one stable model: {models}")

    atom_strings = models[0]["Value"]
    atoms = []
    for atom_string in atom_strings:
        atom = parse_atom(atom_string)
        atoms.append(atom)

    return atoms


def atoms_from_model(stats: Dict):
    if "witnesses" not in stats or len(stats["witnesses"]) == 0:
        return set()

    values = stats["witnesses"][0]["atoms"]
    return set(parse_atom(value) for value in values)

########## ILASP ##########

class ILASPClient:
    title: str
    ilasp_path: str

    def __init__(self, ilasp_path: str = "../ilasp/ILASP", max_literals: int = 3, max_rule_length: int = 5):
        self.ilasp_path = ilasp_path
        self.max_literals = max_literals
        self.max_rule_length = max_rule_length
        self.title = "ilasp"

    def run(self, task_path: str) -> str | None:
        try:
            result = subprocess.run(
                [self.ilasp_path, '--version=4', f'-ml={self.max_literals}', f'--max-rule-length={self.max_rule_length}', '-d', task_path],
                capture_output=True,
                text=True,
                timeout=TIMEOUT
            )
            logging.info(result.stdout)
            return result.stdout
        except subprocess.TimeoutExpired:
            return None

    def run_search_space(self, task_path: str) -> str | None:
        try:
            result = subprocess.run(
                [self.ilasp_path, '--version=4', f'-ml={self.max_literals}', f'--max-rule-length={self.max_rule_length}', '-s', task_path],
                capture_output=True,
                text=True,
                timeout=TIMEOUT
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return None

def format_ilasp_task(original_kb: KnowledgeBase, original_stats: Dict) -> str:
    facts = set(original_kb.facts)
    stable_model_atoms = atoms_from_model(original_stats)

    derived_atoms = stable_model_atoms - facts
    derived_preds = set(a.pred for a in derived_atoms)
    fact_preds = set(a.pred for a in facts)
    all_preds = fact_preds | derived_preds

    all_terms: Set[str] = set()
    for atom in facts:
        all_terms.update(atom.terms)

    lines = []

    for term in sorted(all_terms):
        lines.append(f"#constant(obj, {term}).")
    lines.append("")

    for pred in sorted(derived_preds):
        lines.append(f"#modeh({pred}(var(obj))).")
    lines.append("")

    for pred in sorted(all_preds):
        lines.append(f"#modeb(1, {pred}(var(obj))).")
        lines.append(f"#modeb(1, {pred}(var(obj)), (negative)).")
    lines.append("")

    inclusion = sorted(str(a) for a in derived_atoms)
    exclusion = []
    for pred in sorted(derived_preds):
        for term in sorted(all_terms):
            atom = Atom(pred, (term,))
            if atom not in stable_model_atoms:
                exclusion.append(str(atom))
    context = [f"  {str(f)}." for f in sorted(facts, key=str)]

    lines.append("#pos(eg1, {")
    if inclusion:
        lines.append("  " + ", ".join(inclusion))
    lines.append("}, {")
    if exclusion:
        lines.append("  " + ", ".join(exclusion))
    lines.append("}, {")
    lines.extend(context)
    lines.append("}).")

    return "\n".join(lines)


def hypothesis_space_size(original_kb: KnowledgeBase, original_stats: Dict, max_literals: int) -> int:
    facts = set(original_kb.facts)
    derived_preds = set(a.pred for a in atoms_from_model(original_stats) - facts)
    all_preds = set(a.pred for a in facts) | derived_preds
    n_h = len(derived_preds)
    n_b = 2 * len(all_preds)  # positive + negative modeb for each predicate
    return n_h * sum(comb(n_b, k) for k in range(1, max_literals + 1))


def run_ilasp_for_rules(
    client: ILASPClient,
    original_kb: KnowledgeBase,
    original_stats: Dict,
    task_path: str,
    response_path: str,
    rerun: bool = True
) -> KnowledgeBase:

    if not os.path.exists(response_path) or rerun:
        task_str = format_ilasp_task(original_kb, original_stats)
        write_file(task_path, task_str)
        learned = client.run(task_path)
        if learned is None:
            write_file(response_path, "% TIMED OUT")
            return None
        write_file(response_path, learned)
    else:
        learned = read_file(response_path)
        if learned.strip() == "% TIMED OUT":
            return None

    new_rules = []
    for line in learned.splitlines():
        line = line.strip().rstrip(".")
        if not line or line.startswith("%"):
            continue
        try:
            new_rules.append(parse_rule(line))
        except ValueError:
            continue

    return KnowledgeBase(original_kb.facts, new_rules)


def run_ilasp():

    client = ILASPClient()
    total_num = number_files(f"{DATA_FOLDER}/orig_benchmarks")
    logging.info(f"Run ILASP on {total_num} benchmarks")

    for index in range(total_num):

        orig_benchmark_path = f"{DATA_FOLDER}/orig_benchmarks/benchmark_{index}.lp"
        orig_solution_path = f"{DATA_FOLDER}/orig_solutions/solution_{index}.json"

        task_path = f"{DATA_FOLDER}/ilasp_tasks/task_{index}.las"
        response_path = f"{DATA_FOLDER}/ilasp_responses/response_{index}.las"
        ilasp_benchmarks_path = f"{DATA_FOLDER}/ilasp_benchmarks/benchmark_{index}.lp"
        ilasp_solutions_path = f"{DATA_FOLDER}/ilasp_solutions/solution_{index}.json"

        program = read_file(orig_benchmark_path)
        original_kb = parse_kb(program)
        original_kb = KnowledgeBase(original_kb.facts, [])

        original_stats = read_json(orig_solution_path)
        original_values = atoms_from_model(original_stats)

        h_size = hypothesis_space_size(original_kb, original_stats, client.max_literals)
        logging.info(f"Benchmark {index}: |H| = {h_size}")

        ilasp_kb = run_ilasp_for_rules(client, original_kb, original_stats, task_path, response_path, True)

        if h_size <= SEARCH_SPACE_DEMO_LIMIT:
            search_space_path = f"{DATA_FOLDER}/ilasp_search_spaces/search_space_{index}.las"
            if not os.path.exists(search_space_path):
                os.makedirs(f"{DATA_FOLDER}/ilasp_search_spaces", exist_ok=True)
                search_space = client.run_search_space(task_path)
                write_file(search_space_path, search_space if search_space is not None else "% TIMED OUT")

        if ilasp_kb is None:
            logging.info(f"ILASP timed out on benchmark {index}")
            write_json(ilasp_solutions_path, {"timed_out": True, "hypothesis_space_size": h_size})

            # Use blank to indicate "% TIMED OUT" from ILASP response
            write_file(ilasp_benchmarks_path, "")
            continue

        write_file(ilasp_benchmarks_path, str(ilasp_kb))

        ilasp_stats = run_asp(str(ilasp_kb))
        ilasp_stats["hypothesis_space_size"] = h_size
        write_json(ilasp_solutions_path, ilasp_stats)
        ilasp_values = atoms_from_model(ilasp_stats)

        if original_values != ilasp_values:
            logging.info(f"Original: {original_values}")
            logging.info(f"ILASP:    {ilasp_values}")
        else:
            logging.info(f"Both: {original_values}")


########## Run LLM ##########

def explicit_prompt(facts: str, stable_model: str) -> str:
    return f"""Reconstruct the missing ASP rules given only facts and a target stable model.

Rules must have the form: d{{n}}(Vi) :- [not] d{{m}}(Vj), ...
- Variables may differ across predicates (X, Y, Z, etc.)
- No constants, aggregates, choice rules, or disjunctions
- No new facts (empty body rules)
- Combined with the given facts, rules must yield exactly the target stable model
- Prefer the minimal rule set (fewest rules and literals)

Facts: {facts}
Target stable model: {stable_model}

Output reasoning if helpful, then end with ONLY the rules, one per line.
If no rules are needed: % no additional rules required
If impossible under the schema: % no solution using the allowed rule schema

Example — Facts: d1(a). d1(b). d2(a). | Target: {{d1(a), d1(b), d2(a), d3(a)}}
d3(X) :- d1(X), d2(X).
"""


def feedback_prompt(facts: str, stable_model: str, actual_model: str) -> str:
    return f"""Your rules produced the wrong stable model.

Facts: {facts}
Expected: {stable_model}
Actual:   {actual_model}

Output ONLY the corrected rules as the last lines of your response.
"""


# Run through LLM to get new llm program with the same facts, but generated rules
def run_llm_for_rules(client: LLM, original_kb: KnowledgeBase, messages: list, response_path: str, rerun=True) -> KnowledgeBase:

    logging.info(messages)
    response = get_llm_response(client, messages, response_path, rerun)

    if response is None:
        return None

    messages.append({"role": "assistant", "content": response})

    lines = response.split("\n")
    new_rules = []

    for line_num in range(len(lines)-1, -1, -1):

        rule_str = lines[line_num].replace("`", "").strip()
        if rule_str == "":
            continue

        try:
            rule = parse_rule(rule_str)
            new_rules.append(rule)
        except ValueError:
            break

    if not new_rules:
        logging.warning(f"PARSE ERROR: no rules extracted from response at {response_path!r}")

    return KnowledgeBase(original_kb.facts, new_rules)


def number_files(path: str):

    folder = Path(path)
    return sum(1 for file in folder.iterdir() if file.is_file())


# Run with clingo to see if stable model matches, retrying with feedback on failure
def run_llms(max_iter: int = 3, rerun=True):

    clients = [
        OpenAIClient("gpt-5-mini"),
        OllamaClient("gpt-oss:20b"),
        OllamaClient("qwen3-coder:30b"),
    ]

    total_num = number_files(f"{DATA_FOLDER}/orig_benchmarks")
    logging.info(f"Run {total_num}")

    for client in clients:

        for index in range(total_num):

            logging.info(f"Run {client.title} iteration {index}")

            orig_benchmark_path = f"{DATA_FOLDER}/orig_benchmarks/benchmark_{index}.lp"
            orig_solution_path = f"{DATA_FOLDER}/orig_solutions/solution_{index}.json"

            # Output paths always use index 0 for compatibility with aggregate_results()
            llm_benchmarks_path = f"{DATA_FOLDER}/llm_benchmarks/{client.title}/benchmark_{index}_0.lp"
            llm_solutions_path = f"{DATA_FOLDER}/llm_solutions/{client.title}/solution_{index}_0.json"

            # Original program
            program = read_file(orig_benchmark_path)
            original_kb = parse_kb(program)
            original_kb = KnowledgeBase(original_kb.facts, [])

            # Original statistics
            original_stats = read_json(orig_solution_path)
            original_values = atoms_from_model(original_stats)

            facts_str = ". ".join(str(fact) for fact in original_kb.facts)
            stable_model_str = " ".join(original_stats["witnesses"][0]["atoms"])

            llm_kb = None
            llm_stats = None
            llm_values: Set[Atom] = set()
            timed_out = False
            messages = []

            for run_index in range(max_iter):

                # Each iteration's raw LLM response is cached separately
                llm_responses_path = f"{DATA_FOLDER}/llm_responses/{client.title}/response_{index}_{run_index}.txt"

                if run_index == 0:
                    user_content = explicit_prompt(facts_str, stable_model_str)
                else:
                    if not llm_stats.get("satisfiable"):
                        actual_model_str = "UNSATISFIABLE (no stable model)"
                    elif len(llm_stats["witnesses"]) > 1:
                        actual_model_str = "; ".join(
                            f"Model {w['index']}: " + " ".join(w["atoms"])
                            for w in llm_stats["witnesses"]
                        )
                    else:
                        actual_model_str = " ".join(str(a) for a in llm_values)
                    user_content = feedback_prompt(facts_str, stable_model_str, actual_model_str)

                messages.append({"role": "user", "content": user_content})

                llm_kb = run_llm_for_rules(client, original_kb, messages, llm_responses_path, rerun)

                if llm_kb is None:
                    timed_out = True
                    break

                llm_stats = run_asp(str(llm_kb))
                llm_values = atoms_from_model(llm_stats)

                if original_values == llm_values:
                    logging.info(f"Correct on iteration {run_index}: {llm_values}")
                    break
                else:
                    logging.info(f"Iteration {run_index} failed. Expected: {original_values}, Got: {llm_values}")

            if timed_out:
                logging.info(f"LLM timed out on benchmark {index}")
                write_json(llm_solutions_path, {"timed_out": True})
                continue

            write_file(llm_benchmarks_path, str(llm_kb))
            write_json(llm_solutions_path, llm_stats)

########## Aggregate results ##########

# 7 minutes
ILASP_TIMEOUT_SECONDS = 420
 
_ilasp_phase_re = {
    "time_seconds":                   re.compile(r"%%\s+Total\s*:\s*([\d.]+)s"),
    "ilasp_time_preprocessing":       re.compile(r"%%\s+Pre-processing\s*:\s*([\d.]+)s"),
    "ilasp_time_hypothesis_space_gen":re.compile(r"%%\s+Hypothesis Space Generation\s*:\s*([\d.]+)s"),
    "ilasp_time_conflict_analysis":   re.compile(r"%%\s+Conflict analysis\s*:\s*([\d.]+)s"),
    "ilasp_time_counterexample_search":re.compile(r"%%\s+Counterexample search\s*:\s*([\d.]+)s"),
    "ilasp_time_hypothesis_search":   re.compile(r"%%\s+Hypothesis Search\s*:\s*([\d.]+)s"),
}
_ilasp_iteration_re = re.compile(r"%%\s+Iteration\s+(\d+)\s+%%")
_ilasp_counterexample_re = re.compile(r"a total of (\d+) counterexamples found")

_ILASP_METRICS_NONE = {
    "ilasp_cdilp_iterations": None,
    "ilasp_counterexample_count": None,
    "ilasp_time_preprocessing": None,
    "ilasp_time_hypothesis_space_gen": None,
    "ilasp_time_conflict_analysis": None,
    "ilasp_time_counterexample_search": None,
    "ilasp_time_hypothesis_search": None,
}


def _get_ilasp_metrics(index: int, path: str) -> dict:
    """Parse ILASP response file for timing phases, iteration count, and counterexample count."""
    response_path = f"{path}/ilasp_responses/response_{index}.las"
    if not os.path.exists(response_path):
        return _ILASP_METRICS_NONE.copy()
    content = read_file(response_path)
    if "TIMED OUT" in content:
        return {**_ILASP_METRICS_NONE, "time_seconds": float(ILASP_TIMEOUT_SECONDS)}

    result = {}
    for key, pattern in _ilasp_phase_re.items():
        m = pattern.search(content)
        result[key] = float(m.group(1)) if m else None

    iterations = _ilasp_iteration_re.findall(content)
    result["ilasp_cdilp_iterations"] = max(int(n) for n in iterations) if iterations else None

    m = _ilasp_counterexample_re.search(content)
    result["ilasp_counterexample_count"] = int(m.group(1)) if m else None

    return result


def _get_llm_time(model: str, index: int, path: str) -> float:
    """Sum wall-clock seconds across all iteration response files for one benchmark."""
    total = 0.0
    found_any = False
    run_index = 0
    while True:
        response_path = f"{path}/llm_responses/{model}/response_{index}_{run_index}.txt"
        if not os.path.exists(response_path):
            break
        try:
            data = read_json(response_path)
            if "total_duration" in data:
                # Ollama: nanoseconds to seconds
                total += data["total_duration"] / 1e9
                found_any = True
            elif "wall_time_seconds" in data:
                # OpenAI (new): wall-clock seconds recorded during the call
                total += data["wall_time_seconds"]
                found_any = True
            elif "created_at" in data and "completed_at" in data:
                # OpenAI (existing cached): unix timestamps in seconds
                total += data["completed_at"] - data["created_at"]
                found_any = True
        except Exception:
            pass
        run_index += 1
    return total if found_any else float("nan")


def _get_llm_iter_count(model: str, index: int, path: str) -> int:
    """Count how many LLM iterations were used for one benchmark."""
    count = 0
    while os.path.exists(f"{path}/llm_responses/{model}/response_{index}_{count}.txt"):
        count += 1
    return count


def _get_llm_char_counts(model: str, index: int, path: str) -> dict:
    """Sum thinking and content character/token counts across all iterations for one benchmark."""
    thinking_chars = None
    thinking_tokens = None
    content_chars = None
    run_index = 0
    while True:
        response_path = f"{path}/llm_responses/{model}/response_{index}_{run_index}.txt"
        if not os.path.exists(response_path):
            break
        try:
            data = read_json(response_path)
            if "message" in data:
                # Ollama: thinking and content are plain text
                thinking = data["message"].get("thinking") or ""
                content = data["message"].get("content") or ""
                thinking_chars = (thinking_chars or 0) + len(thinking)
                content_chars = (content_chars or 0) + len(content)
            elif "output" in data:
                # OpenAI: thinking is encrypted; use reasoning_tokens as proxy
                tokens = (data.get("usage") or {}).get("output_tokens_details", {}).get("reasoning_tokens")
                if tokens is not None:
                    thinking_tokens = (thinking_tokens or 0) + tokens
                for item in data.get("output", []):
                    if item.get("type") == "message":
                        for block in item.get("content", []):
                            if block.get("type") == "output_text":
                                content_chars = (content_chars or 0) + len(block.get("text", ""))
        except Exception:
            pass
        run_index += 1
    return {
        "llm_thinking_chars": thinking_chars,
        "llm_thinking_tokens": thinking_tokens,
        "llm_content_chars": content_chars,
    }



def compute_stratification(rules: list) -> dict:
    """Determine whether a set of ASP rules is stratified and compute stratification depth.

    A program is stratified if predicates can be assigned integer strata such that:
    - For every positive body literal: head stratum >= literal stratum
    - For every negative body literal: head stratum >  literal stratum
    (i.e. no cycles through negation).

    Returns a dict with:
      is_stratified      - bool
      stratification_depth - number of distinct strata (0 if non-stratified)
      num_negative_cycles  - number of SCCs that contain a negative edge
    """
    # Build predicate dependency graph.
    # pos_edges[p] = set of predicates q where p has a positive dep on q
    # neg_edges[p] = set of predicates q where p has a negative dep on q
    pos_edges: dict[str, set] = {}
    neg_edges: dict[str, set] = {}
    all_preds: set = set()

    for rule in rules:
        h = rule.head.pred
        all_preds.add(h)
        if h not in pos_edges:
            pos_edges[h] = set()
        if h not in neg_edges:
            neg_edges[h] = set()
        for lit in rule.body:
            q = lit.atom.pred
            all_preds.add(q)
            if q not in pos_edges:
                pos_edges[q] = set()
            if q not in neg_edges:
                neg_edges[q] = set()
            if lit.pos:
                pos_edges[h].add(q)
            else:
                neg_edges[h].add(q)

    if not all_preds:
        return {"is_stratified": True, "stratification_depth": 0, "num_negative_cycles": 0,
                "num_sccs": 0, "max_scc_size": 0, "is_tight": True, "num_positive_cycles": 0}

    # Kosaraju's SCC on the full dependency graph (positive + negative edges combined).
    all_preds_list = list(all_preds)
    combined: dict[str, set] = {p: pos_edges[p] | neg_edges[p] for p in all_preds}
    rev: dict[str, set] = {p: set() for p in all_preds}
    for p, qs in combined.items():
        for q in qs:
            rev[q].add(p)

    # Pass 1: finish-time order
    visited: set = set()
    finish_order: list = []

    def dfs1(node: str) -> None:
        stack = [(node, False)]
        while stack:
            v, done = stack.pop()
            if done:
                finish_order.append(v)
                continue
            if v in visited:
                continue
            visited.add(v)
            stack.append((v, True))
            for w in combined.get(v, ()):
                if w not in visited:
                    stack.append((w, False))

    for p in all_preds_list:
        if p not in visited:
            dfs1(p)

    # Pass 2: assign SCC labels in reverse finish order
    scc_id: dict[str, int] = {}
    num_sccs = 0

    def dfs2(node: str, label: int) -> None:
        stack = [node]
        while stack:
            v = stack.pop()
            if v in scc_id:
                continue
            scc_id[v] = label
            for w in rev.get(v, ()):
                if w not in scc_id:
                    stack.append(w)

    for p in reversed(finish_order):
        if p not in scc_id:
            dfs2(p, num_sccs)
            num_sccs += 1

    # Count SCCs that contain a negative edge (either self-loop or cross-SCC within same SCC).
    num_negative_cycles = 0
    sccs_with_neg_cycle: set = set()
    for p, qs in neg_edges.items():
        for q in qs:
            if scc_id[p] == scc_id[q]:
                sccs_with_neg_cycle.add(scc_id[p])
    num_negative_cycles = len(sccs_with_neg_cycle)

    # SCC sizes
    scc_sizes: dict[int, int] = {}
    for p in all_preds:
        scc_sizes[scc_id[p]] = scc_sizes.get(scc_id[p], 0) + 1
    max_scc_size = max(scc_sizes.values()) if scc_sizes else 0

    # Positive cycles: SCCs with an internal positive edge
    sccs_with_pos_cycle: set = set()
    for p, qs in pos_edges.items():
        for q in qs:
            if scc_id[p] == scc_id[q]:
                sccs_with_pos_cycle.add(scc_id[p])
    num_positive_cycles = len(sccs_with_pos_cycle)
    is_tight = num_positive_cycles == 0

    is_stratified = num_negative_cycles == 0

    if not is_stratified:
        return {
            "is_stratified": False,
            "stratification_depth": 0,
            "num_negative_cycles": num_negative_cycles,
            "num_sccs": num_sccs,
            "max_scc_size": max_scc_size,
            "is_tight": is_tight,
            "num_positive_cycles": num_positive_cycles,
        }

    # Compute strata via topological sort on SCC condensation DAG.
    # scc_stratum[scc] = stratum level (0-indexed)
    scc_stratum: dict[int, int] = {i: 0 for i in range(num_sccs)}

    # Build condensation edges: (src_scc, dst_scc, is_negative)
    cond_in_degree: dict[int, int] = {i: 0 for i in range(num_sccs)}
    cond_pos: dict[int, set] = {i: set() for i in range(num_sccs)}
    cond_neg: dict[int, set] = {i: set() for i in range(num_sccs)}
    seen_cond_edges: set = set()
    for p, qs in pos_edges.items():
        for q in qs:
            s, t = scc_id[p], scc_id[q]
            if s != t and (s, t, False) not in seen_cond_edges:
                seen_cond_edges.add((s, t, False))
                cond_pos[t].add(s)
                cond_in_degree[s] = cond_in_degree.get(s, 0)  # ensure exists
    for p, qs in neg_edges.items():
        for q in qs:
            s, t = scc_id[p], scc_id[q]
            if s != t and (s, t, True) not in seen_cond_edges:
                seen_cond_edges.add((s, t, True))
                cond_neg[t].add(s)

    # BFS/topo over condensation to assign strata
    # stratum of s = max over positive predecessors (same stratum) +
    #                max over negative predecessors (stratum + 1)
    from collections import deque
    in_deg: dict[int, int] = {i: 0 for i in range(num_sccs)}
    succ: dict[int, list] = {i: [] for i in range(num_sccs)}
    for s, t, neg in seen_cond_edges:
        in_deg[s] += 1
        succ[t].append((s, neg))

    queue = deque(i for i in range(num_sccs) if in_deg[i] == 0)
    while queue:
        node = queue.popleft()
        for (child, is_neg) in succ[node]:
            candidate = scc_stratum[node] + (1 if is_neg else 0)
            if candidate > scc_stratum[child]:
                scc_stratum[child] = candidate
            in_deg[child] -= 1
            if in_deg[child] == 0:
                queue.append(child)

    stratification_depth = max(scc_stratum.values()) + 1 if scc_stratum else 1

    return {
        "is_stratified": True,
        "stratification_depth": stratification_depth,
        "num_negative_cycles": 0,
        "num_sccs": num_sccs,
        "max_scc_size": max_scc_size,
        "is_tight": is_tight,
        "num_positive_cycles": num_positive_cycles,
    }


def aggregate_results():

    rows = []

    models = [
        ["original", "orig"],
        ["gpt-oss:20b", "llm"],
        ["gpt-5-mini", "llm"],
        ["qwen3-coder:30b", "llm"],
        ["ilasp", "ilasp"],
    ]

    path = "data"

    total_num = number_files(f"{path}/orig_benchmarks")
    logging.info(f"Aggregate {total_num}")

    for [model, kind] in models:

        for index in range(total_num):

            run_index = 0
            if kind == "orig":
                benchmarks_path = f"{path}/orig_benchmarks/benchmark_{index}.lp"
                solutions_path = f"{path}/orig_solutions/solution_{index}.json"
            elif kind == "ilasp":
                benchmarks_path = f"{path}/ilasp_benchmarks/benchmark_{index}.lp"
                solutions_path = f"{path}/ilasp_solutions/solution_{index}.json"
            else:
                benchmarks_path = f"{path}/llm_benchmarks/{model}/benchmark_{index}_{run_index}.lp"
                solutions_path = f"{path}/llm_solutions/{model}/solution_{index}_{run_index}.json"

            program = read_file(benchmarks_path)
            kb = parse_kb(program)
            metrics = read_json(solutions_path)
            strat = compute_stratification(kb.rules)

            predicates = set()
            possible_terms = set()
            for fact in kb.facts:
                predicates.add(fact.pred)
                if fact.terms:
                    possible_terms.add(fact.terms[0])

            total_pos_literals = sum(1 for rule in kb.rules for lit in rule.body if lit.pos)
            total_neg_literals = sum(1 for rule in kb.rules for lit in rule.body if not lit.pos)

            unique_pos_atoms = len({lit.atom.pred for rule in kb.rules for lit in rule.body if lit.pos})
            unique_neg_atoms = len({lit.atom.pred for rule in kb.rules for lit in rule.body if not lit.pos})
            unique_atoms = len({lit.atom.pred for rule in kb.rules for lit in rule.body})

            body_sizes = [len(r.body) for r in kb.rules]
            max_body_size = max(body_sizes) if body_sizes else 0
            avg_body_size = sum(body_sizes) / len(body_sizes) if body_sizes else 0.0

            neg_body_sizes = [sum(1 for lit in r.body if not lit.pos) for r in kb.rules]
            pos_body_sizes = [sum(1 for lit in r.body if lit.pos) for r in kb.rules]
            avg_neg_body_size = sum(neg_body_sizes) / len(neg_body_sizes) if neg_body_sizes else 0.0
            avg_pos_body_size = sum(pos_body_sizes) / len(pos_body_sizes) if pos_body_sizes else 0.0

            all_atoms = list(kb.facts) + [r.head for r in kb.rules] + [lit.atom for r in kb.rules for lit in r.body]
            max_pred_arity = max((len(a.terms) for a in all_atoms), default=0)

            unique_vars: set = set()
            for r in kb.rules:
                for term in r.head.terms:
                    if term[0].isupper():
                        unique_vars.add(term)
                for lit in r.body:
                    for term in lit.atom.terms:
                        if term[0].isupper():
                            unique_vars.add(term)
            num_unique_vars = len(unique_vars)

            num_recursive_rules = sum(
                1 for r in kb.rules
                if any(lit.atom.pred == r.head.pred for lit in r.body)
            )
            ratio_neg_rules = (
                sum(1 for r in kb.rules if any(not lit.pos for lit in r.body)) / len(kb.rules)
                if kb.rules else 0.0
            )

            solution_atoms = atoms_from_model(metrics)

            llm_timed_out = False
            if kind == "llm":
                time_seconds = _get_llm_time(model, index, path)
                iter_count = _get_llm_iter_count(model, index, path)
                # Cap LLM runs at TIMEOUT, consistent with ILASP. Runs exceeding
                # TIMEOUT are treated as timeout failures by forcing solution_match
                # to False below.
                if pd.notna(time_seconds) and time_seconds > TIMEOUT:
                    time_seconds = float(TIMEOUT)
                    llm_timed_out = True
                ilasp_metrics = _ILASP_METRICS_NONE.copy()
                llm_chars = _get_llm_char_counts(model, index, path)
            elif kind == "ilasp":
                ilasp_metrics = _get_ilasp_metrics(index, path)
                time_seconds = ilasp_metrics.pop("time_seconds") or 0.0
                iter_count = 1
                llm_chars = {"llm_thinking_chars": None, "llm_thinking_tokens": None, "llm_content_chars": None}
            else:
                time_seconds = 0.0
                iter_count = 0
                ilasp_metrics = _ILASP_METRICS_NONE.copy()
                llm_chars = {"llm_thinking_chars": None, "llm_thinking_tokens": None, "llm_content_chars": None}

            solver = metrics.get("statistics", {})

            search_space_path = f"{path}/ilasp_search_spaces/search_space_{index}.las"
            if os.path.exists(search_space_path):
                with open(search_space_path) as _f:
                    search_space_lines = sum(1 for _ in _f)
            else:
                search_space_lines = None

            new_row = {
                "benchmark_name": model,
                "example_index": index,
                "num_predicates": len(predicates),
                "num_possible_terms": len(possible_terms),
                "num_facts": len(kb.facts),
                "num_rules": len(kb.rules),
                "total_pos_literals": total_pos_literals,
                "total_neg_literals": total_neg_literals,
                "unique_pos_atoms": unique_pos_atoms,
                "unique_neg_atoms": unique_neg_atoms,
                "unique_atoms": unique_atoms,
                "is_stratified": strat["is_stratified"],
                "stratification_depth": strat["stratification_depth"],
                "num_negative_cycles": strat["num_negative_cycles"],
                "num_sccs": strat["num_sccs"],
                "max_scc_size": strat["max_scc_size"],
                "is_tight": strat["is_tight"],
                "num_positive_cycles": strat["num_positive_cycles"],
                "max_body_size": max_body_size,
                "avg_body_size": avg_body_size,
                "avg_neg_body_size": avg_neg_body_size,
                "avg_pos_body_size": avg_pos_body_size,
                "max_predicate_arity": max_pred_arity,
                "num_unique_vars": num_unique_vars,
                "num_recursive_rules": num_recursive_rules,
                "ratio_neg_rules": ratio_neg_rules,
                "solution_atoms": solution_atoms,
                "num_solution_atoms": len(solution_atoms),
                "num_derived_atoms": len(solution_atoms) - len(kb.facts),
                "rules": kb.rules,
                "facts": kb.facts,
                "time_seconds": time_seconds,
                "llm_timed_out": llm_timed_out,
                "iter_count": iter_count,
                "solver_conflicts": solver.get("conflicts"),
                "solver_choices": solver.get("choices"),
                "solver_time_solve": solver.get("time_solve"),
                "solver_ground_atoms": solver.get("atoms"),
                "solver_ground_rules": solver.get("rules"),
                "hypothesis_space_size": metrics.get("hypothesis_space_size"),
                "search_space_line_count": search_space_lines,
                **ilasp_metrics,
                **llm_chars,
            }

            rows.append(new_row)

    analysis_df = pd.DataFrame(rows).astype({
        "benchmark_name": "string",
        "llm_timed_out": "bool",
        "example_index": "int64",
        "num_predicates": "int64",
        "num_possible_terms": "int64",
        "num_facts": "int64",
        "num_rules": "int64",
        "total_pos_literals": "int64",
        "total_neg_literals": "int64",
        "unique_pos_atoms": "int64",
        "unique_neg_atoms": "int64",
        "unique_atoms": "int64",
        "solution_atoms": "object",
        "num_solution_atoms": "int64",
        "num_derived_atoms": "int64",
        "rules": "object",
        "facts": "object",
        "time_seconds": "float64",
        "iter_count": "int64",
        "solver_conflicts": "float64",
        "solver_choices": "float64",
        "solver_time_solve": "float64",
        "solver_ground_atoms": "float64",
        "solver_ground_rules": "float64",
        "hypothesis_space_size": "float64",
        "search_space_line_count": "float64",
        "ilasp_cdilp_iterations": "float64",
        "ilasp_counterexample_count": "float64",
        "ilasp_time_preprocessing": "float64",
        "ilasp_time_hypothesis_space_gen": "float64",
        "ilasp_time_conflict_analysis": "float64",
        "ilasp_time_counterexample_search": "float64",
        "ilasp_time_hypothesis_search": "float64",
        "llm_thinking_chars": "float64",
        "llm_thinking_tokens": "float64",
        "llm_content_chars": "float64",
        "num_sccs": "int64",
        "max_scc_size": "int64",
        "num_positive_cycles": "int64",
        "max_body_size": "int64",
        "avg_body_size": "float64",
        "avg_neg_body_size": "float64",
        "avg_pos_body_size": "float64",
        "max_predicate_arity": "int64",
        "num_unique_vars": "int64",
        "num_recursive_rules": "int64",
        "ratio_neg_rules": "float64",
    })

    original_df = analysis_df[analysis_df["benchmark_name"] == "original"]
    llm_df = analysis_df[analysis_df["benchmark_name"] != "original"]

    joined_df = llm_df.merge(
        original_df,
        on="example_index",
        how="inner",
        suffixes=("_llm", "_original"),
    )

    joined_df["solution_match"] = (
        joined_df["solution_atoms_llm"] == joined_df["solution_atoms_original"]
    )
    # LLM runs exceeding TIMEOUT are treated as failures regardless of answer set.
    joined_df.loc[joined_df["llm_timed_out_llm"] == True, "solution_match"] = False

    joined_df["rules_match"] = (
        joined_df["rules_llm"] == joined_df["rules_original"]
    )

    llm_models = {"gpt-oss:20b", "gpt-5-mini", "qwen3-coder:30b"}

    def _later_success(row):
        # solution_{index}_0.json always holds the final result; response_{index}_1.txt
        # only exists when iteration 0 failed, so its presence + solution_match=True
        # means the LLM succeeded on a later iteration.
        if not row["solution_match"]:
            return False
        if row["benchmark_name_llm"] not in llm_models:
            return False
        path = f"data/llm_responses/{row['benchmark_name_llm']}/response_{row['example_index']}_1.txt"
        return os.path.exists(path)

    joined_df["later_success"] = joined_df.apply(_later_success, axis=1)

    def _atom_metrics(row):
        llm = row["solution_atoms_llm"]
        orig = row["solution_atoms_original"]
        intersection = len(llm & orig)
        precision = intersection / len(llm) if llm else 0.0
        recall = intersection / len(orig) if orig else 0.0
        denom = precision + recall
        f1 = 2 * precision * recall / denom if denom > 0 else 0.0
        return pd.Series({
            "atom_precision": precision,
            "atom_recall": recall,
            "atom_f1": f1,
            "num_extra_atoms": len(llm - orig),
            "num_missing_atoms": len(orig - llm),
        })

    joined_df[["atom_precision", "atom_recall", "atom_f1", "num_extra_atoms", "num_missing_atoms"]] = joined_df.apply(_atom_metrics, axis=1)

    num_true = joined_df["rules_match"].sum()

    logging.info(num_true)
    logging.info(joined_df)

    return joined_df


def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s\n%(message)s"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["generate", "ilasp", "llms", "aggregate"])
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    if args.stage == "generate":
        generate_benchmarks()
    elif args.stage == "ilasp":
        run_ilasp()
    elif args.stage == "llms":
        run_llms(max_iter=args.max_iter, rerun=args.rerun)
    elif args.stage == "aggregate":
        aggregate_results()


if __name__ == "__main__":
    main()
