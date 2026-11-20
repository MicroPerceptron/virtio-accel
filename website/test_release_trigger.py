import unittest
from release_trigger import eligible_run, published_to_pypi, published_release


class ReleaseTriggerTests(unittest.TestCase):
    def event(self, **changes):
        run = dict(conclusion='success', head_repository={'full_name': 'owner/repo'},
                   event='release', name='Release Python SDK')
        run.update(changes)
        return {'workflow_run': run, 'repository': {'full_name': 'owner/repo'}}

    def test_successful_release_and_manual_workflows(self):
        for name in ('Publish crates', 'Release Python SDK'):
            for event in ('release', 'workflow_dispatch'):
                self.assertTrue(eligible_run(self.event(name=name, event=event)))

    def test_failed_unrelated_fork_and_pr_runs_are_excluded(self):
        for change in ({'conclusion': 'failure'}, {'conclusion': 'cancelled'},
                       {'name': 'Python CI'}, {'event': 'pull_request'},
                       {'head_repository': {'full_name': 'fork/repo'}}):
            with self.subTest(change=change):
                self.assertFalse(eligible_run(self.event(**change)))

    def test_skipped_rust_job_on_python_tags_does_not_refresh(self):
        for status in ('success', 'skipped', 'failure'):
            self.assertEqual(published_release('Publish crates', [
                dict(name='publish-crates-io', conclusion=status)]), status == 'success')

    def test_only_successful_production_step_triggers_refresh(self):
        for conclusion in ('success', 'failure', 'skipped'):
            for step_name in ('Publish to PyPI', 'Publish to TestPyPI'):
                for step_conclusion in ('success', 'failure', 'skipped'):
                    jobs = [dict(name='publish', conclusion=conclusion, steps=[
                        dict(name=step_name, conclusion=step_conclusion)])]
                    self.assertEqual(published_to_pypi(jobs), conclusion == 'success'
                                     and step_name == 'Publish to PyPI' and step_conclusion == 'success')
        self.assertFalse(published_to_pypi([]))
        self.assertFalse(published_to_pypi([dict(name='publish', conclusion='skipped')]))


if __name__ == '__main__':
    unittest.main()
