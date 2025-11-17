from dataclasses import dataclass
from typing import List, Set, Tuple, Dict, Optional, Iterable
import random
import os
import json
import re
import time
import clingo
from abc import ABC, abstractmethod
from openai import OpenAI
from pathlib import Path
from ollama import Client
import pandas as pd

########## ASP Models ##########

@dataclass(frozen=True)
class Atom:
    pred: str
    terms: Tuple[str, ...]

    def is_ground(self) -> bool:
        return all(not arg[0].islower() for arg in self.terms)

    def __str__(self):
        terms_str = ", ".join(str(arg) for arg in self.terms)
        return f"{self.pred}({terms_str})"


@dataclass(frozen=True)
class Literal:
    pos: bool
    atom: Atom

    def __str__(self):
        return f"{"" if self.pos else "not "}{self.atom}"


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


def parse_rule(rule_string: str) -> Rule:
    rule_string = rule_string.strip().rstrip(".")

    if ":-" in rule_string:
        head_str, body_str = map(str.strip, rule_string.split(":-", 1))
        lits = []
    
        for part in body_str.split(","):

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
        key_obj = json.loads(read_file("key.json"))
        key = key_obj["openai"]
        self.client = OpenAI(api_key=key)
        self.title = title


    def send_prompt(self, prompt: str) -> str:

        # frequency_penalty=1.0,
        # presence_penalty=1.0,
        # temperature=1.0,

        response = self.client.responses.create(
            model=self.title,
            input=[{
                "role": "user", 
                "content": prompt
            }],
            # max_output_tokens=1000
        )

        full_response = json.loads(response.model_dump_json())
        print(json.dumps(full_response, indent=2))
        response_content = full_response["output"][1]["content"][0]["text"]

        return [full_response, response_content]


    def get_content(self, full_response: str):
        
        return full_response["output"][1]["content"][0]["text"]


class OllamaClient(LLM):
    client: Client
    title: str

    def __init__(self, title):
        self.client = Client()
        self.title = title


    def send_prompt(self, prompt: str) -> str:

        response = self.client.chat(
            model=self.title, 
            messages=[{
                'role': 'user',
                'content': prompt
            }]
        )

        full_response = json.loads(response.json())
        response_content = full_response["message"]["content"]

        return [full_response, response_content]
    

    def get_content(self, full_response):

        return full_response["message"]["content"]


def get_llm_response(client: LLM, prompt: str, response_path: str, rerun: bool):

    print('Run', client.title)

    # Check if output is already present, don't run if it is
    if not os.path.exists(response_path) or rerun:

        [full_response, response_content] = client.send_prompt(prompt)

        print(full_response)
        write_json(response_path, full_response)

    else:
        full_response = read_json(response_path)
        response_content = client.get_content(full_response)

    print('Response:')
    print(response_content)
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

# Generator creates instances and rules
# d0(o0).
# d0(o1).
# d1(o1).
# d2(X) :- d0(X), not d1(X).

# Example base
# vehicle(bike).
# vehicle(car).
# slow_moving(bike).
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
        head = Atom(head_predicate, "X")
        remaining_pred = list(predicates - set([head_predicate]))

        body = set()
        for literal_id in range(config.num_literals):

            # Select an unused atom
            to_ground_pred = random.sample(remaining_pred, 1)[0]
            atom = Atom(to_ground_pred, "X")

            # Ensure at least one positive and 
            if config.num_literals <= config.num_neg_literals:
                raise RuntimeError('Too many negative literals')

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
    while (len(model_stats["witnesses"]) != 1 or 
           len(model_stats["warnings"]) != 0 or 
           len(kb.facts) == len(model_stats["witnesses"][0]['atoms'])):

        if len(model_stats["witnesses"]) != 1:
            print(f"Not only 1 stable model: {model_stats["witnesses"]}")

        elif len(model_stats["warnings"]) != 0:
            print(f"Has warnings: {model_stats["warnings"]}")

        elif len(kb.facts) == len(model_stats["witnesses"][0]['atoms']):
            print(f"Needs more atoms: {model_stats["witnesses"][0]['atoms']}")

        kb = generate_example(config)
        model_stats = run_asp(str(kb))

    benchmark_path = f"data/orig_benchmarks/benchmark_{index}.lp"
    solution_path = f"data/orig_solutions/solution_{index}.json"

    write_file(benchmark_path, str(kb))
    write_json(solution_path, model_stats)


def generate_benchmarks():

    index = 0
    for example_number in range(5):
        for num_predicates in [4, 5]:
            for num_possible_terms in [4, 5]:
                for num_facts in [4]:
                    for num_rules in [1, 2]:
                        for num_literals in [1, 2]:
                            for num_neg_literals in [0, 1]:

                                if num_literals <= num_neg_literals:
                                    continue

                                config = ModelConfig(
                                    example_number,
                                    num_predicates,
                                    num_possible_terms,
                                    num_facts,
                                    num_rules,
                                    num_literals,
                                    num_neg_literals
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


def read_json(path_str: str) -> str:
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
    ctl.add("base", [], program)

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
    if len(stats['witnesses']) == 0:
        return set()

    values = stats['witnesses'][0]['atoms']
    return set(parse_atom(value) for value in values)

########## Run LLM ##########

def simple_prompt(facts: str, stable_model: str):
    return f"""
You are given the following ASP facts:
{facts}
And rules of the form
predicate_0(X) :- predicate_1(X), not predicate_2(X), etc.

That results in the stable model:
{stable_model}

Deduce minimal set of these rules where each predicate can be one of [d0, d1, d2, etc.]
Write the rules as the final lines of output.
"""


def explicit_prompt(facts: str, stable_model: str):

    return f"""
You are given:

1. An ASP program's facts (all rules have been removed).
2. A target stable model that the original (complete) program produced.

Your task is to reconstruct the missing rules.
Each rule must conform strictly to this schematic form (for any predicate symbol d{{n}} and arity 1):

d{{n}}(X) :- L1, L2, ..., Lk.

where each literal Li is either d{{m}}(X) or not d{{m}}(X) for some predicate d{{m}}.

Constraints:

- Allowed predicates in rule bodies: only d{{m}}(X) (positive) or not d{{m}}(X) (default negation).
- Allowed head predicates: only d{{n}}(X) (arity 1).
- Variables: use only the single variable X (appearing in the head for safety).
- No constants except those appearing in the given facts.
- No aggregates, choice rules, disjunctions, or integrity constraints.
- No facts (rules with empty bodies); only the provided facts are to remain facts.
- The reconstructed rules, when combined with the given facts, must yield exactly the provided stable model under standard stable-model semantics.
- Prefer the minimal set of rules (fewest total rules and literals) that achieves this.
- If multiple minimal sets exist, output any one valid minimal set.

Input:

Facts (only):
{facts}

Target stable model:
{stable_model}

Output instructions (IMPORTANT):

- You may include a brief "Reasoning:" section to explain your derivation.
- After your reasoning, end your message with ONLY the rules, one per line, with no commentary, headers, or trailing text below them.
- The last non-empty lines of your entire response must be exactly the rules in ASP syntax.

If no rules are needed to obtain the target stable model from the facts, end with a single line:
% no additional rules required

If it is impossible to obtain exactly the target stable model using only the allowed rule schema, end with a single line:
% no solution using the allowed rule schema

Procedure you should follow (do not print these steps):
1) Identify all d{{n}}/1 predicates appearing in the facts and in the target stable model.
2) Hypothesize candidate rules of the allowed form that, together with the facts, yield exactly the target stable model.
3) Test and prune candidates to ensure the result is stable and minimal.
4) Output any reasoning you wish, then finish with ONLY the final rules, one per line.

Example (illustrative only; do not reuse for the actual task):

Facts:
d1(a). d2(a).

Target stable model:
{{ d1(a), d2(a), d3(a) }}

Acceptable output shape:
Reasoning: From d1(a) and d2(a) we must derive d3(a); minimal positive rule suffices.

d3(X) :- d1(X), d2(X).
"""


# Run through LLM to get new llm program with the same facts, but generated rules
def run_llm_for_rules(client: LLM, original_kb: KnowledgeBase, original_stats: Dict, response_path: str, rerun=False) -> KnowledgeBase:

    facts = ". ".join(str(fact) for fact in original_kb.facts)
    stable_model = " ".join(original_stats["witnesses"][0]['atoms'])
    
    prompt = simple_prompt(facts, stable_model)
    print(prompt)

    response = get_llm_response(client, prompt, response_path, rerun)

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

    llm_kb = KnowledgeBase(original_kb.facts, new_rules)
    return llm_kb


# Run with clingo to see if stable model matches
def run_llms():

    clients = [
        OllamaClient("gpt-oss:20b"),
        # OllamaClient("qwen2.5-coder:32b"),
        OpenAIClient("gpt-5-mini"),
        OllamaClient("qwen3-coder:30b"),
        # OllamaClient("deepseek-r1:14b")
    ]

    for client in clients:

        for index in range(120):

            run_index = 0

            orig_benchmark_path = f"data/orig_benchmarks/benchmark_{index}.lp"
            orig_solution_path = f"data/orig_solutions/solution_{index}.json"

            llm_responses_path = f"data/llm_responses/{client.title}/response_{index}_{run_index}.txt"
            llm_benchmarks_path = f"data/llm_benchmarks/{client.title}/benchmark_{index}_{run_index}.lp"
            llm_solutions_path = f"data/llm_solutions/{client.title}/solution_{index}_{run_index}.json"

            # Original program
            program = read_file(orig_benchmark_path)
            original_kb = parse_kb(program)
            original_kb = KnowledgeBase(original_kb.facts, [])

            # Original statistics
            original_stats = json.loads(read_file(orig_solution_path))
            original_values = atoms_from_model(original_stats)

            # Run LLM
            llm_kb = run_llm_for_rules(client, original_kb, original_stats, llm_responses_path, False)
            write_file(llm_benchmarks_path, str(llm_kb))

            # Run LLM code on clingo and save
            llm_stats = run_asp(str(llm_kb))
            write_json(llm_solutions_path, llm_stats)
            llm_values = atoms_from_model(llm_stats)

            # If fails, retry saying what is incorrect
            # If passes or out of retries, save results 

            if original_values != llm_values:
                print(f"Original: {original_values}")
                print(f"LLM:      {llm_values}")
            else:
                print(f"Both: {original_values}")

########## Aggregate results ##########

def aggregate_results():

    analysis_df = pd.DataFrame(columns=[
        "benchmark_name",
        "example_index",
        "num_predicates",
        "num_possible_terms",
        "num_facts",
        "num_rules",
        "total_pos_literals",
        "total_num_neg_literals",
        "solution_atoms",
        "num_solution_atoms",
        "rules",
        "facts"
    ])

    analysis_df = analysis_df.astype({
        "benchmark_name": "string",
        "example_index": "int64",
        "num_predicates": "int64",
        "num_possible_terms": "int64",
        "num_facts": "int64",
        "num_rules": "int64",
        "total_pos_literals": "int64",
        "total_num_neg_literals": "int64",
        "solution_atoms": "object",
        "num_solution_atoms": "int64",
        "rules": "object",
        "facts": "object"
    })

    models = [
        ["original", False],
        ["gpt-oss:20b", True ],
        ["gpt-5-mini", True ],
        ["qwen3-coder:30b", True]
        # ["deepseek-r1:32b", True ]
    ]

    for [ model, is_llm ] in models:

        for index in range(120):

            run_index = 0
            benchmarks_path = f"data/llm_benchmarks/{model}/benchmark_{index}_{run_index}.lp" if is_llm else f"data/orig_benchmarks/benchmark_{index}.lp"
            solutions_path = f"data/llm_solutions/{model}/solution_{index}_{run_index}.json" if is_llm else f"data/orig_solutions/solution_{index}.json" 

            program = read_file(benchmarks_path)
            kb = parse_kb(program)
            metrics = read_json(solutions_path)

            predicates = set()
            possible_terms = set()
            for fact in kb.facts:
                predicates.add(fact.pred)
                possible_terms.add(fact.terms[0])

            pos_literals = set()
            neg_literals = set()
            for rule in kb.rules:
                for literal in rule.body:
                    if literal.pos:
                        pos_literals.add(literal.atom)
                    else:
                        neg_literals.add(literal.atom)

            solution_atoms = atoms_from_model(metrics)

            new_row = {
                "benchmark_name": model,
                "example_index": index,
                "num_predicates": len(predicates),
                "num_possible_terms": len(possible_terms),
                "num_facts": len(kb.facts),
                "num_rules": len(kb.rules),
                "total_pos_literals": len(pos_literals),
                "total_num_neg_literals": len(neg_literals),
                "solution_atoms": solution_atoms,
                "num_solution_atoms": len(solution_atoms),
                "rules": kb.rules,
                "facts": kb.facts
            }

            analysis_df.loc[len(analysis_df)] = new_row

    original_df = analysis_df[analysis_df["benchmark_name"] == "original"]
    llm_df = analysis_df[analysis_df["benchmark_name"] != "original"]

    joined_df = llm_df.merge(
        original_df,
        on="example_index",
        how="inner",
        suffixes=("_llm", "_original")
    )

    joined_df["solution_match"] = (
        joined_df["solution_atoms_llm"] == joined_df["solution_atoms_original"]
    )

    joined_df["rules_match"] = (
        joined_df["rules_llm"] == joined_df["rules_original"]
    )

    num_true = joined_df["solution_match"].sum()

    print(num_true)
    print(joined_df)
    

def main():

    # generate_benchmarks()

    # run_llms()

    aggregate_results()


if __name__ == "__main__":
    main()
