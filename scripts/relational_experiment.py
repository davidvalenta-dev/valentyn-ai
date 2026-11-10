"""Reproduce frozen relational-policy evaluation; never fits on evaluation data."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from valentyn.cli import provenance
from valentyn.engine import Engine
from valentyn.factorized_evaluation import evaluate
from valentyn.memory import Memory
from valentyn.proof_data import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(args.protocol.read_text())
    for path, checksum in protocol['checkpoints'].items():
        if hashlib.sha256((Path(path)/'weights.pt').read_bytes()).hexdigest() != checksum:
            raise SystemExit('Frozen checkpoint changed')
    data = load_dataset(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    memory = Memory(root/'.valentyn/relational/frozen-results.sqlite3')
    from valentyn.verification_cache import VerificationCache
    verifier = VerificationCache(root)
    selected = Path(protocol['primary_checkpoint'])
    arms = []
    for seed in protocol['training_seeds']:
        for beam in (1,4):
            arms.append((f'cost-seed{seed}-beam{beam}', Path(f'.valentyn/relational/cost-seed{seed}'), beam, None, None))
    for label, checkpoint in (('imitation',Path('.valentyn/relational/seed7')),
                              ('old-gru',Path('models/valentyn-state')),
                              ('untrained',selected/'initial')):
        for beam in (1,4):
            arms.append((f'{label}-beam{beam}', checkpoint, beam, None, None))
    for control in ('ordered','structural'):
        for beam in (1,4):
            arms.append((f'{control}-beam{beam}', selected, beam, control, None))
    for control, label in ((None,'random'),('random-closed','closed-random')):
        for seed in protocol['random_seeds']:
            for beam in (1,4):
                arms.append((f'{label}-seed{seed}-beam{beam}',selected,beam,control,seed))
    for label, checkpoint, beam, control, seed in arms:
        output = args.output/f'{label}.json'
        if args.skip_existing and output.is_file():
            continue
        report = evaluate(root,memory,checkpoint,data,beam=beam,nodes=protocol['node_budget'],
                          control=control,random_seed=seed,verifier=verifier)
        report.update({'arm':label,'provenance':provenance(root),
                       'protocol_sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                       'verification_cache':{'hits_cumulative':verifier.hits,'compilations_cumulative':verifier.compilations}})
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(label,report['solved'],report['tasks'],report['nodes'],flush=True)
    for budget in (200,5000):
        for strategy in ('focused','forward','split'):
            output=args.output/f'symbolic-{strategy}-{budget}.json'
            if args.skip_existing and output.is_file():
                continue
            engine=Engine(root,memory,nodes=budget,depth=24,timeout=5)
            engine.verifier=verifier
            start=time.monotonic()
            results=[engine.attempt(t['formula'],strategy,task_id=t['id'],split='frozen-v2') for t in data['test']]
            report={'kind':'symbolic_control','arm':output.stem,'dataset_digest':data['digest'],
                    'tasks':len(results),'solved':sum(r['status']=='verified' for r in results),
                    'max_nodes':budget,'strategy':strategy,'results':results,
                    'inference_seconds':sum(r['elapsed'] for r in results),
                    'total_seconds':time.monotonic()-start,'provenance':provenance(root),
                    'note':'Symbolic ticks include forward saturation; not equivalent to typed search expansions. Teacher selection favors this solver.'}
            output.write_text(json.dumps(report,indent=2)+'\n')
            print(output.stem,report['solved'],report['tasks'],flush=True)
    for path, checksum in protocol['checkpoints'].items():
        if hashlib.sha256((Path(path)/'weights.pt').read_bytes()).hexdigest() != checksum:
            raise SystemExit('Checkpoint changed during evaluation')


if __name__=='__main__':
    main()
