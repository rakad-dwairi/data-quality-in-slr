"""Offline quality-gate checks; no real model or retrieval is constructed."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from validate_fast_local_model import quality_checks
from continue_local_review import checked_quality
from slr_pipeline import screening
from slr_pipeline.common import write_json


class FastModelTests(unittest.TestCase):
    def row(self, values, kind='primary', decision='uncertain'):
        data={'decision':decision,'study_kind':kind,'rationale':'Synthetic fixture.',
              'criteria':[{'criterion':n,'value':values.get(n,'unclear'),
                           'source_attribution':'own_study', 'explanation':'Synthetic evidence.',
                           'quote':'Synthetic fixture.', 'passage_id':'ABSTRACT'} for n in screening.CORE_CRITERIA]}
        return {'title':'Fixture', **screening._validate(data,{'ABSTRACT':'Synthetic fixture.'},
                 'title_abstract', list(screening.CORE_CRITERIA), {})}

    def test_secondary_classification_uses_unchanged_core_priority(self):
        row=self.row({'investigated_data_property':'no','primary_research':'no'},'secondary','exclude')
        self.assertEqual(row['ai_decision'],'exclude')
        self.assertTrue(quality_checks([row],{'Fixture':'secondary'})['passed'])
        row['ai_study_kind']='primary'
        self.assertFalse(quality_checks([row],{'Fixture':'secondary'})['passed'])

    def test_positive_false_discard_or_unrecognised_context_fails(self):
        for row in [self.row({'recommender_context':'no'},decision='exclude'),self.row({'primary_research':'yes'})]:
            self.assertFalse(quality_checks([row],{'Fixture':'retain'})['passed'])

    def test_unrelated_control_must_be_excluded(self):
        row=self.row({'primary_research':'yes'})
        self.assertFalse(quality_checks([row],{'Fixture':'out_of_scope'})['passed'])

    def test_failure_report_cannot_prepare_continuation(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);source=p/'source';quality=p/'quality'
            write_json(source/'manifest.json',{'config':{'protocol':{}}})
            write_json(quality/'quality_report.json',{'adoptable':False})
            write_json(quality/'provenance.json',{})
            write_json(quality/'configuration.json',{})
            with patch('requests.sessions.Session.request',side_effect=AssertionError('No network allowed')):
                with self.assertRaisesRegex(ValueError,'did not pass'):
                    checked_quality(source,quality)

class PacingResumeTests(unittest.TestCase):
    def test_interrupted_budget_is_charged_once_and_finished_spend_survives_restart(self):
        from continue_local_review import EarlyReviewBudget
        from slr_pipeline.common import read_json
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'budget.json'
            with patch('continue_local_review.time.time',return_value=100), patch('continue_local_review.time.monotonic',return_value=200):
                budget=EarlyReviewBudget(path,floor=10)
                budget.begin('SYNTHETIC')
            with patch('continue_local_review.time.time',return_value=140):
                resumed=EarlyReviewBudget(path)
            self.assertEqual(resumed.spent,50)
            self.assertIsNone(read_json(path)['in_progress'])
            with patch('continue_local_review.time.time',return_value=160), patch('continue_local_review.time.monotonic',return_value=300):
                again=EarlyReviewBudget(path)
                self.assertEqual(again.spent,50)
                again.begin('SECOND')
            with patch('continue_local_review.time.monotonic',return_value=315):
                again.finish()
            self.assertEqual(EarlyReviewBudget(path).spent,65)
            self.assertEqual(EarlyReviewBudget(path,floor=80).spent,80)
