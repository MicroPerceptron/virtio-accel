"""Regression checks for keeping build caches and hidden files out of the site."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build
import check


class TreeTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.book = self.root / 'book'
        self.write(self.book / 'index.html')
        self.write(self.book / 'python/api/index.html')

    def write(self, path, text=''):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def problems(self):
        with patch.object(check, 'BOOK', self.book):
            return check.check_unpublished_files()


class UnpublishedFileCheckTests(TreeTestCase):
    def test_tree_of_ordinary_files_passes(self):
        self.write(self.book / 'python/api/objects.inv')
        self.write(self.book / 'python/api/_static/pygments.css')
        self.assertEqual(self.problems(), [])

    def test_sphinx_doctree_cache_is_rejected_once(self):
        self.write(self.book / 'python/api/.doctrees/environment.pickle')
        self.write(self.book / 'python/api/.doctrees/index.doctree')
        self.assertEqual(self.problems(), [
            'hidden directory in the published tree: python/api/.doctrees/'])

    def test_hidden_files_are_rejected_anywhere_in_the_tree(self):
        for rel in ('.nojekyll', 'python/api/.buildinfo', 'api/.lock'):
            self.write(self.book / rel)
        self.assertEqual(self.problems(), [
            'hidden file in the published tree: .nojekyll',
            'hidden file in the published tree: api/.lock',
            'hidden file in the published tree: python/api/.buildinfo'])

    def test_build_cache_is_rejected_without_a_hidden_parent(self):
        for rel in ('python/api/doctrees/environment.pickle',
                    'python/api/doctrees/index.doctree', 'assets/MODEL.PKL'):
            self.write(self.book / rel)
        problems = self.problems()
        self.assertEqual(len(problems), 3)
        self.assertTrue(all('build cache' in item for item in problems))


class BuildOutputTests(TreeTestCase):
    def test_sphinx_doctrees_stay_outside_the_site(self):
        calls = []

        def run(command, **kwargs):
            doctrees = Path(command[command.index('-d') + 1])
            self.assertTrue(doctrees.is_dir())
            # What Sphinx would have written, had it been left its default.
            self.write(doctrees / 'environment.pickle')
            calls.append((command[command.index('-b') + 1], doctrees))

        with patch.object(build, 'BOOK', self.book), patch.object(
                build.subprocess, 'run', side_effect=run):
            build.build_python_docs()
        self.assertEqual([builder for builder, _ in calls], ['html', 'doctest'])
        html, doctest = (doctrees for _, doctrees in calls)
        # The doctest run reads the documents the HTML run parsed.
        self.assertEqual(html, doctest)
        self.assertFalse(html.resolve().is_relative_to(self.book.resolve()))
        self.assertFalse(html.exists())
        self.assertEqual(self.problems(), [])

    def test_named_generator_leftovers_are_removed(self):
        for rel in build.GENERATOR_LEFTOVERS:
            self.write(self.book / rel)
        with patch.object(build, 'BOOK', self.book):
            build.remove_generator_leftovers()
            build.remove_generator_leftovers()
        self.assertEqual(self.problems(), [])

    def test_unknown_hidden_file_is_left_for_the_check_to_report(self):
        self.write(self.book / 'python/api/.mystery')
        with patch.object(build, 'BOOK', self.book):
            build.remove_generator_leftovers()
        self.assertEqual(self.problems(), [
            'hidden file in the published tree: python/api/.mystery'])

    def test_copied_trees_leave_hidden_files_behind(self):
        target_doc = self.root / 'target/doc'
        assets = self.root / 'assets'
        self.write(target_doc / '.lock')
        self.write(target_doc / 'virtio_accel/index.html')
        self.write(assets / '.DS_Store')
        self.write(assets / 'logo/.DS_Store')
        self.write(assets / 'logo/logo.png')
        with patch.object(build, 'BOOK', self.book), patch.object(
                build, 'TARGET_DOC', target_doc), patch.object(
                build, 'ASSETS', assets), patch.object(build.subprocess, 'run'):
            build.build_rustdoc()
            build.copy_assets()
        self.assertTrue((self.book / 'api/virtio_accel/index.html').exists())
        self.assertTrue((self.book / 'api/index.html').exists())
        self.assertTrue((self.book / 'assets/logo/logo.png').exists())
        self.assertEqual(self.problems(), [])


if __name__ == '__main__':
    unittest.main()
