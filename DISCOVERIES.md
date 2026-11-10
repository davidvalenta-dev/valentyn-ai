# Findings and limitations

The practical finding is that local typed action prediction transfers better than whole-term generation on this synthetic curriculum. Strong random-search controls account for much of that transfer. This suggests investigating search guidance and compositional representations; it does not establish a novel architecture or a mathematical discovery.

Adding intermediate forward derivations and alternative-action cost supervision produced the strongest current result: 190/192 on a fresh frozen suite with beam 4, compared with 148/192 for the prior learned policy. The 5000-node symbolic control still solves 192/192, and the action language remains narrow propositional logic.

Portable GQA and absorbed MLA reduce retained cache tensors as expected. No proof-quality improvement was established. The unfused delta/GQA hybrid was slower at this scale. No MoE benefit has been demonstrated, so MoE remains deferred.

All accepted mathematical outputs are elementary propositions. No Millennium Prize Problem has been solved by Valentyn. The Lean library proves soundness of a formal rule system, and the verifier checks emitted terms; neither proves correctness of the entire implementation or a general intelligence claim.
