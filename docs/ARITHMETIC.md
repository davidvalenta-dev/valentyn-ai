# Arithmetic and algebra track

The first arithmetic track is intentionally independent of the propositional proof generator. `src/valentyn/arithmetic.py` contains a bounded expression parser and evaluator for natural-number expressions using `+`, `-`, `*`, variables and parentheses. An identity is accepted only after a pinned Lean 4.24 compiler checks the generated theorem.

The current curriculum has seven verified examples: three closed calculations and four variable identities whose equality is definitionally reducible by Lean's natural-number evaluator. The model is not yet trained on this track, and the curriculum does not include general commutativity, inequalities, induction, fields, or polynomial normalization. Those require a richer proof language and theorem library.

The next research step is to represent algebraic proof states as typed rewrite goals, train action selection on both successful and failed rewrites, and compare the learned policy to normalization and exhaustive rewrite controls. Every proposed identity must remain independently checked by Lean; numerical agreement alone is not proof.
