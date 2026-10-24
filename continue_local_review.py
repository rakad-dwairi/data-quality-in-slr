#!/usr/bin/env python3
"""Checkpoint continuation after a passed local TA model comparison; no searches.

Use --prepare only after stopping the original workers. The original scientific
artifacts stay in place. Use --run to resume the prepared continuation.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import os
from pathlib import Path
import shutil
import tempfile
import time

from slr_pipeline import cli, recover
from slr_pipeline.common import digest, file_hash, read_csv, read_json, run_lock, utcnow, write_json
from slr_pipeline.resume import _stopped_source
from local_retention import retain_title_abstract
from pending_title_abstract import screen_remaining

ROOT = Path(__file__).resolve().parent
FILES = ('continue_local_review.py', 'pending_title_abstract.py', 'local_retention.py',
         'local_continuation_review.py', 'local_source_citations.py',
         'early_local_review.py', 'validate_fast_local_model.py')


class EarlyReviewBudget:
    """Persist pacing spend; conservatively charge interrupted review time."""
    def __init__(self, path, floor=0.0):
        self.path = Path(path)
        state = read_json(self.path, {})
        spent = float(state.get('spent_seconds', 0))
        if state.get('in_progress'):
            spent += max(0, time.time() - state['in_progress']['started_epoch'])
        self.spent = max(spent, floor)
        self._save()

    def _save(self, in_progress=None):
        write_json(self.path, {'updated_utc': utcnow(), 'spent_seconds': self.spent,
                              'in_progress': in_progress})

    def begin(self, record_id):
        self.started = time.monotonic()
        self._save({'record_id': record_id, 'started_epoch': time.time()})

    def finish(self):
        self.spent += max(0, time.monotonic() - self.started)
        self._save()


def review_spend_floor(output, started_utc):
    """Recover first-version spending from its own saved retrieval timestamps.

    Unfinished attempts include downtime conservatively; they never earn fresh
    bootstrap credit. Donor reports are outside this output and excluded.
    """
    earliest = datetime.fromisoformat(started_utc).timestamp()
    total = 0.0
    for folder in (Path(output) / 'reports').glob('*'):
        attempts = folder / '04_full_text/retrieval_attempts.jsonl'
        starts = []
        if attempts.is_file():
            import json
            starts = [datetime.fromisoformat(json.loads(line)['attempted_utc']).timestamp()
                      for line in attempts.read_text().splitlines() if line.strip()]
        result = folder / 'result.json'
        end = result.stat().st_mtime if result.exists() else time.time()
        total += max(0, end - max(earliest, min(starts) if starts else earliest))
    return total


def checked_quality(source, quality):
    manifest = read_json(source / 'manifest.json')
    report = read_json(quality / 'quality_report.json')
    provenance = read_json(quality / 'provenance.json')
    config = read_json(quality / 'configuration.json')
    amended = read_json(quality / 'quality_gate_correction.json', {})
    if amended:
        if (amended.get('runner_sha256') != file_hash(ROOT / 'validate_fast_local_model.py')
                or amended.get('assessments_sha256') != file_hash(quality / 'assessments.json')
                or amended.get('report_sha256') != file_hash(quality / 'quality_report.json')
                or amended.get('previous_runner_sha256') != provenance.get('runner_sha256')
                or amended.get('inference_performed') is not False
                or amended.get('audit_hashes') != {p.name:file_hash(p) for p in sorted((quality / 'checkpoints/model_audit').glob('*.json'))}):
            raise ValueError('The documented offline quality-gate correction changed')
    if (not report.get('adoptable') or not report.get('passed') or not report.get('speed_gate_passed')
            or report.get('errors') or report.get('records', 0) < 12):
        raise ValueError('The candidate did not pass the bounded quality and speed gates')
    if (provenance.get('source_manifest_sha256') != file_hash(source / 'manifest.json')
            or provenance.get('candidate_config_sha256') != file_hash(quality / 'configuration.json')
            or provenance.get('source_inputs_sha256') != file_hash(quality / 'source_inputs.csv')
            or (not amended and provenance.get('runner_sha256') != file_hash(ROOT / 'validate_fast_local_model.py'))
            or provenance.get('source_citation_adapter_sha256') != file_hash(ROOT / 'local_source_citations.py')
            or provenance.get('protocol_sha256') != digest(manifest['config']['protocol'])
            or config['protocol'] != manifest['config']['protocol']):
        raise ValueError('The passed candidate/protocol/source provenance changed')
    model = config['model']
    if model.get('allow_remote') is not False or model.get('base_url') not in {'http://127.0.0.1:11434', 'http://127.0.0.1:11435'}:
        raise ValueError('Only the tested loopback-only model is allowed')
    audits = [read_json(p) for p in (quality / 'checkpoints/model_audit').glob('*.json')]
    identities = {(a.get('model'), a.get('model_digest')) for a in audits}
    if len(audits) != report['records'] or len(identities) != 1:
        raise ValueError('Quality check lacks a single complete model identity')
    name, model_digest = next(iter(identities))
    if name != model['name'] or not model_digest:
        raise ValueError('Quality check used a different model')
    report = {**report, 'candidate_model_digest': model_digest}
    return config, report


def prepare(source, quality, target, source_early):
    source, quality, target, source_early = map(lambda p: Path(p).resolve(), (source, quality, target, source_early))
    if target.exists() or any(target == p or p in target.parents or target in p.parents for p in (source, quality, source_early)):
        raise ValueError('Choose a new separate continuation directory')
    config, report = checked_quality(source, quality)
    config.update(run_directory=str(target), _project_dir=str(ROOT))
    config['search']['enabled'] = False
    config['screening'].update(limit=0, retry_errors=False)
    with _stopped_source(source), _stopped_source(source_early):
        manifest, changes, _ = recover._verify_manifest(source, ROOT)
        if changes:
            raise ValueError('The original core implementation must be unchanged')
        state, stages = recover._verify_stages(source)
        records = read_csv(source / '02_deduplication/records.csv')
        retained = retain_title_abstract(source, records)
        retained_ids = {x['record_id'] for x in retained['rows']}
        pending_ids = [r['record_id'] for r in records if r['record_id'] not in retained_ids]
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.local-model-handoff-', dir=target.parent))
        inventory = []
        try:
            for name in ('01_retrieval', '02_deduplication'):
                recover._copy_tree(source / name, staging / name, staging, inventory)
            recover._copy_checkpoints(source / 'checkpoints', staging / 'checkpoints', staging, inventory)
            recover._initialize_current(config, staging, ROOT)
            write_json(staging / 'stage_state.json', {**{s: copy.deepcopy(state[s]) for s in stages},
                                                     'screen': {'valid': False}, 'fulltext': {'valid': False}})
            write_json(staging / 'continuation_config.json', {k:v for k,v in config.items() if not k.startswith('_')})
            write_json(staging / '03_title_abstract/retained_rows.json', retained['rows'])
            write_json(staging / '03_title_abstract/retained_ledger.json', retained['ledger'])
            write_json(staging / 'retention_history.json', retained['history'])
            write_json(staging / 'pending_ids.json', pending_ids)
            shutil.copytree(quality, staging / 'quality_check')
            root_hashes = {}
            for name in FILES:
                recover._copy_file(ROOT / name, staging / 'coordinator_snapshot' / name, staging, inventory)
                root_hashes[name] = file_hash(ROOT / name)
            write_json(staging / 'continuation_origin.json', {
                'created_utc': utcnow(), 'source_run': str(source), 'source_early_review': str(source_early),
                'source_manifest_sha256': file_hash(source / 'manifest.json'),
                'source_early_manifest_sha256': file_hash(source_early / 'manifest.json'),
                'source_fingerprint': manifest['fingerprint'], 'protocol_sha256': digest(config['protocol']),
                'cohort_records': len(records), 'retained_assessments': len(retained['rows']),
                'historical_checkpoint_rows': retained['historical_rows'], 'remaining_records': len(pending_ids),
                'retained_rows_sha256': file_hash(staging / '03_title_abstract/retained_rows.json'),
                'retained_ledger_sha256': file_hash(staging / '03_title_abstract/retained_ledger.json'),
                'pending_ids_sha256': file_hash(staging / 'pending_ids.json'),
                'root_source_hashes': root_hashes, 'candidate_quality_report': report,
                'database_requests': 0, 'paid_ai_calls': 0, 'fulltext_model_policy': 'Original strong local model and original blinded prompts retained',
                'note': 'Mixed-model TA continuation, with original per-assessment provenance. No target count or human approvals invented.'})
            write_json(staging / 'copied_evidence.json', inventory)
            recover._verify_stages(staging)
            staging.rename(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return read_json(target / 'continuation_origin.json')


def run(target):
    # Imports happen only after preparation and frozen-source verification.
    from local_continuation_review import ReviewContinuation
    import early_local_review as early
    target = Path(target).resolve()
    origin = read_json(target / 'continuation_origin.json')
    for name, expected in origin['root_source_hashes'].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError('Frozen continuation source changed: ' + name)
    for relative, key in [('03_title_abstract/retained_rows.json', 'retained_rows_sha256'),
                          ('03_title_abstract/retained_ledger.json', 'retained_ledger_sha256'),
                          ('pending_ids.json', 'pending_ids_sha256')]:
        if file_hash(target / relative) != origin[key]:
            raise ValueError('Frozen retention input changed: ' + relative)
    config = read_json(target / 'continuation_config.json'); config['_project_dir'] = str(ROOT)
    cli.initialize(config, target)
    records = read_csv(target / '02_deduplication/records.csv')
    retained = read_json(target / '03_title_abstract/retained_rows.json')
    ledger = read_json(target / '03_title_abstract/retained_ledger.json')
    if digest(config['protocol']) != origin['protocol_sha256'] or len(records) != origin['cohort_records']:
        raise ValueError('The original protocol/cohort changed')
    service = ROOT / 'runs/local_completion_service'
    def status(phase, **extra):
        data = {'updated_utc': utcnow(), 'pid': os.getpid(), 'phase': phase,
                'run_directory': str(target), 'model': config['model']['name'],
                'retained_assessments': len(retained), 'database_searches': 0, 'paid_ai_calls': 0, **extra}
        write_json(target / 'progress.json', data)
        write_json(service / 'completion_progress.json', data)
    with run_lock(target):
        reviewer = ReviewContinuation(Path(origin['source_early_review']), target / 'evidence_review')
        initial_results = reviewer.results
        baseline_count = len(initial_results)
        budget = EarlyReviewBudget(target / 'early_review_budget.json',
                                   review_spend_floor(target / 'evidence_review', origin['created_utc']))
        median = origin['candidate_quality_report']['candidate_median_seconds']
        def review_callback(completed):
            new = sum(r.get('screening_status') == 'assessed' and r.get('retained_from_run') is None for r in completed)
            allowance = 90 + new * median * .15
            if budget.spent >= allowance:
                return
            by_id = {r['record_id']:r for r in records}
            done = {r['summary']['record_id'] for r in reviewer.results}
            entries = [{'record_id':r['record_id'], 'cache_key':r.get('cache_key',''),
                        'result':r, 'created_at':r.get('assessed_utc','')} for r in completed]
            queue = early.shortlist(entries, by_id, done, early.KNOWN_ITEM_TITLES)
            if queue:
                item = queue[0]
                status('early_fulltext_review', record_id=item['record_id'])
                budget.begin(item['record_id'])
                try:
                    reviewer.review(by_id[item['record_id']], item['cache_key'])
                finally:
                    budget.finish()
                reviewer.export()
            progress = read_json(target / '03_title_abstract/progress.json', {})
            status('title_abstract_screening', **{k:v for k,v in progress.items() if k not in {'phase','updated_utc'}})
        try:
            status('title_abstract_screening')
            rows = screen_remaining(records, retained, config, target, retained_ledger=ledger,
                                    batch_size=64, after_batch=review_callback,
                                    expected_model_digest=origin['candidate_quality_report']['candidate_model_digest'])
            ta, _, _, _ = cli.reviewed_stages(config, target, ta=copy.deepcopy(rows), ft=[])
            queue = [r for r in ta if cli.advances_to_fulltext(r)]
            queue.sort(key=lambda r:(not bool(r.get('oa_pdf_url')), r.get('ai_decision') != 'include', r['record_id']))
            status('fulltext_screening', fulltext_queue=len(queue), retained_early_reports=baseline_count)
            done = {r['summary']['record_id'] for r in reviewer.results}
            for i, record in enumerate(queue):
                if record['record_id'] not in done:
                    reviewer.review(record, record.get('cache_key',''))
                    done.add(record['record_id'])
                if i % 16 == 0:
                    reviewer.export()
                    status('fulltext_screening', fulltext_processed=len(done), fulltext_queue=len(queue))
            reviewer.export()
            firsts = [r['first'] for r in reviewer.results if r.get('first')]
            cli.snapshot_stage(target, firsts, 'full_text')
            cli.end_stage(target, 'fulltext', firsts)
            cli.export_results(config, target, ta=copy.deepcopy(rows), ft=copy.deepcopy(firsts))
            status('ai_processing_finished_human_review_required', reports_reviewed=len(reviewer.results),
                   double_pass_candidates=sum(r['summary']['review_status']=='double_pass_candidate' for r in reviewer.results))
        except BaseException as exc:
            status('needs_attention', error=type(exc).__name__ + ': ' + str(exc))
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--source', type=Path)
    p.add_argument('--quality', type=Path)
    p.add_argument('--source-early', type=Path)
    p.add_argument('--run', type=Path, required=True)
    args=p.parse_args()
    if args.prepare:
        if not all((args.source,args.quality,args.source_early)):
            p.error('--prepare requires --source, --quality and --source-early')
        result=prepare(args.source,args.quality,args.run,args.source_early)
        print({k:result[k] for k in ('retained_assessments','remaining_records','historical_checkpoint_rows')})
    else:
        run(args.run)


if __name__=='__main__':
    main()
