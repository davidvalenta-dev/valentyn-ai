# Research roadmap

These are contingent research phases, not predictions or delivery dates. Advancement requires measured evidence; adding modules or increasing model size does not itself satisfy a milestone.

| Phase | Experimentally verifiable milestone | Discovery needed to advance | Current standing |
|---|---|---|---|
| 1. Working mathematical agent | Reproducible reasoning loop, independent proof checking, multiple strategies, archived failures, held-out comparison, no accepted invalid results in adversarial tests | A task representation and feedback signal that support learning beyond a good fixed strategy | Initial propositional version implemented; cross-domain mathematics and research-grade evaluation remain |
| 2. Improves its reasoning algorithms | New search procedures beat the best fixed baseline on at least three frozen task families after charging proposal/training/selection cost; no material retained-domain regressions across five seeds | Transferable algorithmic improvements rather than validation-set tuning | Numeric policy evolution exists; zero observed promotions in the first run; executable code evolution is deferred |
| 3. Contributes novel mathematics | At least one independently reviewed result absent from the literature, with faithful formal statement, reproducible proof, and explicit attribution of human contributions | Conjecture selection, useful abstractions, and novelty assessment extending beyond elementary identities | Conjecture generation is implemented, global novelty checking is not |
| 4. Generalizes scientific discovery | A fixed learning procedure produces independently validated improvements in mathematics plus two distinct scientific/computational domains; cross-domain ablations establish transfer | Shared representations and learning signals that survive domain changes and empirical noise | Proposed; no current evidence |
| 5. Investigates broadly superhuman reasoning | Consistent superiority over strong expert/tool baselines on fresh, independently designed research tasks, measured with complete compute and human assistance accounting | Durable generalization, long-horizon planning, stable continual learning, and improved discovery efficiency | Research objective; no capability claim |
| 6. Major open mathematical problems | Expert-guided subproblem decomposition, faithful formalizations, accepted intermediate theorems, and ultimately independently assessed solutions under the relevant problem criteria | Problem-specific mathematical breakthroughs plus reliable research-level reasoning | No Millennium problem solution or serious partial solution is claimed |

## Immediate sequence

1. Preserve the small verified implementation and recorded negative results.
2. Freeze a larger generated suite with size/family holdouts and independently check its generator. Compare best constant, contextual, and portfolio strategies over five seeds.
3. Profile end-to-end CPU, compiler time, disk, and peak memory on the actual M2. Improve the dominant cost before adding a large model.
4. Add per-node proof-state traces and exact verified theorem reuse. Test whether memory amortizes its costs before attempting abstraction discovery.
5. Extend the typed problem language to natural-number arithmetic and finite combinatorial constructions, each with separate formal semantics and checkers.
6. Implement certified macro extraction and genuinely generated search-controller programs in an OS sandbox; preserve the evaluator and test tasks outside the candidate's write access.
7. Only then test small learned value functions, graph recurrence, or optional pretrained mathematical proposals against the stronger symbolic baselines.

## Hardware and funding transitions

The first release performs small CPU experiments with no model weights. Train the strategy learner from scratch; do not confuse that with training a general-purpose intelligence. Small graph/value models can also start from random initialization. Existing open proof models can later supply candidates or traces when that makes a hypothesis cheaper to test, while remaining outside the trust boundary.

Move to external compute when measurements show that a promising experiment is limited by throughput, not because larger hardware is assumed to create a missing idea. Candidate funding evidence should include code, reproducibility artifacts, a controlled positive or informative negative result, and a costed next experiment. Any use of paid APIs, cloud GPUs, private data, or collaborators requires an explicit budget and access arrangements before charges or external communication.

For Millennium problems, first choose a domain with mathematical collaborators, a formal-library assessment, and useful intermediate questions. A proof checker validates a formal statement; independent mathematical judgement is necessary to establish that the statement addresses the actual problem.
