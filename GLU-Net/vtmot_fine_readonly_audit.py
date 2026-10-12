"""Read-only verification of the VTMOT A1 fine-adaptation checkpoint and val CSVs.

Only the four validation sequences are opened. No model selection or test access.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from datasets.vtmot_video import VTMOTVideos
from roadscene_coarse import sha256_file


STAGES = ('coarse16', 'local32', 'local64', 'final512')


def rows_at(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    keyed = {(row['sequence'], row['stem']): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f'duplicate validation frame in {path}')
    return keyed


def weighted(rows, field):
    count = sum(int(row['valid_pixels']) for row in rows)
    if count <= 0:
        raise ValueError('no valid validation pixels')
    return sum(float(row[field]) * int(row['valid_pixels']) for row in rows) / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fine-dir', type=Path, required=True)
    parser.add_argument('--coarse-dir', type=Path, required=True)
    parser.add_argument('--split-file', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-inference', action='store_true')
    parser.add_argument('--pretrained', type=Path)
    parser.add_argument('--coarse-checkpoint', type=Path)
    parser.add_argument('--a1-checkpoint', type=Path)
    args = parser.parse_args()
    if args.output.resolve() == args.fine_dir.resolve() or args.output.resolve() == args.coarse_dir.resolve():
        parser.error('audit output must be separate from training output')
    split = json.loads(args.split_file.read_text(encoding='utf-8'))
    if len(split['eval']) != 4 or len(split['test']) != 6 or set(split['eval']) & set(split['test']):
        raise ValueError('expected the locked four-val/six-test disjoint split')
    videos = VTMOTVideos(args.data_root, args.split_file, 'eval')
    if set(videos.sequences) != set(split['eval']) or any(len(stems) != 200 for stems in videos.sequences.values()):
        raise ValueError('validation sequence/frame manifest changed')
    expected = {(name, stem) for name, stems in videos.sequences.items() for stem in stems}
    if len(expected) != 800:
        raise ValueError('validation manifest must contain 800 frames')

    fine_path = args.fine_dir / 'best_fine_adapted_a1.pth'
    coarse_path = args.coarse_dir / 'best_adapted_a1.pth'
    fine_sha = sha256_file(fine_path)
    coarse_sha = sha256_file(coarse_path)
    fine_report = json.loads((args.fine_dir / 'report_val.json').read_text(encoding='utf-8'))
    coarse_report = json.loads((args.coarse_dir / 'report_val.json').read_text(encoding='utf-8'))
    fine_checkpoint = torch.load(fine_path, map_location='cpu', weights_only=False)
    if (fine_checkpoint.get('recipe') != 'vtmot_a1_fine_only_v1' or
            fine_checkpoint.get('stage') != 'fine' or
            fine_checkpoint.get('source_adapted_sha256') != coarse_sha or
            fine_checkpoint.get('split_sha256') != sha256_file(args.split_file) or
            fine_checkpoint.get('test_accessed') is not False or
            fine_report.get('test_accessed') is not False or
            coarse_report.get('test_accessed') is not False or
            fine_report.get('best_fine_sha256') != fine_sha or
            fine_report.get('source_adapted_sha256') != coarse_sha or
            fine_report.get('selected_epoch') != fine_checkpoint.get('epoch')):
        raise ValueError('checkpoint, source weights, split or report provenance mismatch')
    history = json.loads((args.fine_dir / 'history.json').read_text(encoding='utf-8'))
    if history and (fine_report.get('stopped_at_epoch') != history[-1]['epoch'] or
                    fine_checkpoint['epoch'] > history[-1]['epoch']):
        raise ValueError('best checkpoint and training history disagree')
    fine = rows_at(args.fine_dir / 'vtmot_val_per_frame.csv')
    coarse = rows_at(args.coarse_dir / 'vtmot_val_per_frame.csv')
    paired = rows_at(args.fine_dir / 'vtmot_val_paired.csv')
    if set(fine) != expected or set(coarse) != expected or set(paired) != expected:
        raise ValueError('missing, extra or test validation frame in per-frame CSV')

    paired_rows = []
    for name, stems in videos.sequences.items():
        for stem in stems:
            key = (name, stem)
            old, new, saved = coarse[key], fine[key], paired[key]
            frame = videos.input_frame(name, stem)
            _, valid = videos.flow_truth(name, stem, frame['original_hw'])
            count = int(valid.sum())
            if count != int(old['valid_pixels']) or count != int(new['valid_pixels']) or count != int(saved['valid_pixels']):
                raise ValueError(f'GT valid-mask count differs at {name}/{stem}')
            old_epe = float(old['final512_epe_512px'])
            new_epe = float(new['final512_epe_512px'])
            if (abs(old_epe - float(saved['coarse_adapted_epe_512px'])) > 1e-5 or
                    abs(new_epe - float(saved['fine_adapted_epe_512px'])) > 1e-5):
                raise ValueError(f'paired CSV values differ at {name}/{stem}')
            paired_rows.append({'sequence': name, 'stem': stem, 'valid_pixels': count,
                                'before_final_epe_512px': old_epe,
                                'after_final_epe_512px': new_epe,
                                'change_512px': new_epe - old_epe})

    full_before = weighted(list(coarse.values()), 'final512_epe_512px')
    full_after = weighted(list(fine.values()), 'final512_epe_512px')
    reported_before = coarse_report['adapted']['all']['stage_epe_512px']['final512']
    reported_after = fine_report['fine_adapted']['all']['stage_epe_512px']['final512']
    if (abs(full_before - reported_before) > 0.01 or
            abs(full_after - reported_after) > 0.01 or
            abs(full_after - fine_checkpoint['val_final_epe_512px']) > 0.01):
        raise ValueError('full validation EPE differs from checkpoint or reports')
    per_sequence = {}
    for name in videos.sequences:
        old_rows = [coarse[(name, stem)] for stem in videos.sequences[name]]
        new_rows = [fine[(name, stem)] for stem in videos.sequences[name]]
        pairs = [r for r in paired_rows if r['sequence'] == name]
        per_sequence[name] = {
            'frames': 200, 'valid_pixels': sum(r['valid_pixels'] for r in pairs),
            'before_final_epe_512px': weighted(old_rows, 'final512_epe_512px'),
            'after_final_epe_512px': weighted(new_rows, 'final512_epe_512px'),
            'regressed_frames': sum(r['change_512px'] > 0 for r in pairs),
            'improved_frames': sum(r['change_512px'] < 0 for r in pairs),
            'worst_after': sorted(pairs, key=lambda r: r['after_final_epe_512px'], reverse=True)[:3],
            'largest_regressions': sorted(pairs, key=lambda r: r['change_512px'], reverse=True)[:3],
        }
    report = {
        'verified': True, 'test_accessed': False,
        'protocol': 'four locked VTMOT validation sequences, 800 frames, same independent GT-valid mask, 512x512 final flow pixel units',
        'best_checkpoint_path': str(fine_path.resolve()), 'best_checkpoint_sha256': fine_sha,
        'source_coarse_checkpoint_sha256': coarse_sha,
        'selected_epoch': fine_checkpoint['epoch'], 'optimizer_steps': fine_checkpoint['optimizer_steps'],
        'all': {'frames': 800,
                'valid_pixels': sum(r['valid_pixels'] for r in paired_rows),
                'before_final_epe_512px': full_before,
                'after_final_epe_512px': full_after,
                'improved_frames': sum(r['change_512px'] < 0 for r in paired_rows),
                'regressed_frames': sum(r['change_512px'] > 0 for r in paired_rows),
                'worst_after': sorted(paired_rows, key=lambda r: r['after_final_epe_512px'], reverse=True)[:10],
                'largest_regressions': sorted(paired_rows, key=lambda r: r['change_512px'], reverse=True)[:10]},
        'per_sequence': per_sequence,
    }
    if args.verify_inference:
        if not all((args.pretrained, args.coarse_checkpoint, args.a1_checkpoint)):
            parser.error('--verify-inference requires pretrained, coarse and A1 weights')
        from vtmot_a1_adapt import evaluate_vtmot, install_mutual_guard, seed_all
        from vtmot_a1_fine_adapt import load_start
        if (fine_checkpoint['pretrained_sha256'] != sha256_file(args.pretrained) or
                fine_checkpoint['coarse_sha256'] != sha256_file(args.coarse_checkpoint) or
                fine_checkpoint['source_a1_sha256'] != sha256_file(args.a1_checkpoint)):
            raise ValueError('fine checkpoint base weight hashes differ')
        seed_all()
        install_mutual_guard()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        recipe = SimpleNamespace(pretrained=args.pretrained,
                                 coarse_checkpoint=args.coarse_checkpoint,
                                 a1_checkpoint=args.a1_checkpoint,
                                 coarse_adapt_checkpoint=coarse_path,
                                 split_file=args.split_file)
        model, _ = load_start(recipe, device)
        model.load_state_dict(fine_checkpoint['model_state_dict'], strict=True)
        model.base.train_coarse_encoder = False
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        independent, summary = evaluate_vtmot(model, videos, device)
        discrepancy = max(abs(float(row['final512_epe_512px']) -
                              float(fine[(row['sequence'], row['stem'])]
                                    ['final512_epe_512px'])) for row in independent)
        inference_epe = summary['all']['stage_epe_512px']['final512']
        if len(independent) != 800 or abs(inference_epe-full_after) > .01 or discrepancy > .1:
            raise ValueError(f'independent inference mismatch: {inference_epe}, '
                             f'max frame difference {discrepancy}')
        report['independent_inference'] = {
            'frames': len(independent), 'final_flow_epe_512px': inference_epe,
            'maximum_frame_epe_difference_512px': discrepancy,
            'same_input_split_and_GT_mask': True}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    with (args.output / 'per_frame_paired.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
