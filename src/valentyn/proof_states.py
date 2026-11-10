"""Typed proof construction environment and supervision extracted from proof terms.

This module supplies inference rules, not a search policy. Model or random-policy
experiments choose every action. Final terms still require independent Lean checks.
"""

from dataclasses import dataclass
import re

from .logic import Formula

RULES = ("intro", "pair", "left", "right", "absurd")
ACTIONS = (*RULES, *(f"exact:{i}" for i in range(32)),
           *(f"apply:{i}" for i in range(32)), *(f"cases:{i}" for i in range(32)))


@dataclass(frozen=True)
class Term:
    head: str
    args: tuple = ()

    def lean(self) -> str:
        if self.head == "lambda":
            return f"(fun {self.args[0]} => {self.args[1].lean()})"
        if self.head == "call":
            return "(" + " ".join(arg.lean() for arg in self.args) + ")"
        return self.head


def parse_term(source: str) -> Term:
    parts = re.findall(r"=>|[()]|[A-Za-z][A-Za-z0-9_.]*|\S", source)
    index = 0

    def expression() -> Term:
        nonlocal index
        if index >= len(parts):
            raise ValueError("Incomplete proof term")
        token = parts[index]
        index += 1
        if token == "(":
            if index < len(parts) and parts[index] == "fun":
                index += 1
                variable = parts[index]
                index += 1
                if parts[index] != "=>":
                    raise ValueError("Malformed lambda")
                index += 1
                body = expression()
                result = Term("lambda", (variable, body))
            else:
                args = []
                while index < len(parts) and parts[index] != ")":
                    args.append(expression())
                if not args:
                    raise ValueError("Empty application")
                result = args[0] if len(args) == 1 else Term("call", tuple(args))
            if index >= len(parts) or parts[index] != ")":
                raise ValueError("Unclosed proof term")
            index += 1
            return result
        if token in (")", "=>"):
            raise ValueError("Unexpected proof token")
        return Term(token)

    result = expression()
    if index != len(parts):
        raise ValueError("Trailing proof tokens")
    return result


def flatten(term: Term) -> tuple[Term, list[Term]]:
    if term.head != "call":
        return term, []
    head, earlier = flatten(term.args[0])
    return head, earlier + list(term.args[1:])


def saturate(context: tuple[tuple[Formula, Term], ...]) -> tuple[tuple[Formula, Term], ...]:
    """Conjunction projection only; no recursive proof search or implication search."""
    facts = list(context)
    known = {formula for formula, _ in facts}
    for formula, term in facts:
        if formula.op == "and":
            for subgoal, projection in zip(formula.args, ("And.left", "And.right")):
                if subgoal not in known:
                    known.add(subgoal)
                    facts.append((subgoal, Term("call", (Term(projection), term))))
        if len(facts) > 32:
            raise ValueError("Proof-state context exceeds 32 facts")
    return tuple(facts)


def premises(function: Formula, goal: Formula) -> list[Formula] | None:
    required = []
    while function.op == "imp":
        required.append(function.args[0])
        function = function.args[1]
        if function == goal:
            return required
    return None


def available(goal: Formula, context) -> list[int]:
    context = saturate(context)
    choices = []
    if goal.op == "imp":
        choices.append(ACTIONS.index("intro"))
    if goal.op == "and":
        choices.append(ACTIONS.index("pair"))
    if goal.op == "or":
        choices.extend((ACTIONS.index("left"), ACTIONS.index("right")))
    if goal.op != "False" and any(f.op == "False" or premises(f, Formula("False")) for f, _ in context):
        choices.append(ACTIONS.index("absurd"))
    for index, (formula, _) in enumerate(context):
        if formula == goal:
            choices.append(ACTIONS.index(f"exact:{index}"))
        if premises(formula, goal):
            choices.append(ACTIONS.index(f"apply:{index}"))
        if formula.op == "or":
            choices.append(ACTIONS.index(f"cases:{index}"))
    # True has a dedicated trivial proof, represented as an exact fact in context.
    return choices


def with_truth(context):
    if not any(formula.op == "True" for formula, _ in context):
        context = (*context, (Formula("True"), Term("True.intro")))
    return saturate(tuple(context))


def extract_steps(goal: Formula, proof: str) -> list[dict]:
    """Bidirectional decomposition of verified teacher terms into local decisions."""
    steps = []

    def walk(target: Formula, term: Term, context) -> None:
        context = with_truth(context)
        label = None
        children = []
        for index, (formula, _) in enumerate(context):
            if formula == target:
                label = f"exact:{index}"
                break
        head, args = flatten(term)
        if label is None and term.head == "lambda" and target.op == "imp":
            variable, body = term.args
            label = "intro"
            children = [(target.args[1], body,
                         (*context, (target.args[0], Term(variable))))]
        elif label is None and head.head == "And.intro" and target.op == "and":
            label = "pair"
            children = [(g, t, context) for g, t in zip(target.args, args)]
        elif label is None and head.head in ("Or.inl", "Or.inr") and target.op == "or":
            direction = 0 if head.head == "Or.inl" else 1
            label = ("left", "right")[direction]
            children = [(target.args[direction], args[0], context)]
        elif label is None and head.head == "False.elim":
            # Teach the direct false-context action when available.
            if any(f.op == "False" for f, _ in context):
                label = "absurd"
            else:
                label = "absurd"
                children = [(Formula("False"), args[0], context)]
        elif label is None and head.head == "Or.elim":
            source, left, right = args
            for index, (formula, evidence) in enumerate(context):
                if formula.op == "or" and evidence == source:
                    label = f"cases:{index}"
                    remainder = tuple((f, t) for f, t in context if f != formula)
                    if left.head != "lambda" or right.head != "lambda":
                        raise ValueError("Expected explicit case lambdas")
                    children = [(formula.args[0], left, remainder), (formula.args[1], right, remainder)]
                    children = [(target, branch.args[1], (*remainder, (antecedent, Term(branch.args[0]))))
                                for antecedent, branch, _ in children]
                    break
        elif label is None:
            for index, (formula, evidence) in enumerate(context):
                needed = premises(formula, target)
                # Curried applications may have nested application nodes.
                evidence_head, evidence_args = flatten(evidence)
                if needed and head == evidence_head and args[:len(evidence_args)] == evidence_args:
                    supplied = args[len(evidence_args):]
                    if len(supplied) == len(needed):
                        label = f"apply:{index}"
                        children = [(g, t, context) for g, t in zip(needed, supplied)]
                        break
        if label is None:
            raise ValueError(f"No local action for {target.lean()}: {term.lean()}")
        action = ACTIONS.index(label)
        if action not in available(target, context):
            raise ValueError("Teacher action is not legal")
        steps.append({"goal": target, "context": context, "action": action})
        for subgoal, child, child_context in children:
            walk(subgoal, child, child_context)

    walk(goal, parse_term(proof), ())
    return steps
