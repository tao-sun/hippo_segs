import unittest
from pathlib import Path

from snn_fptt import limit_subject_dirs


class SubjectLimitTest(unittest.TestCase):
    def test_limit_subject_dirs_returns_first_n_subjects(self):
        subjects = [Path(f"subject-{i}") for i in range(5)]

        limited = limit_subject_dirs(subjects, 3, "fold 1")

        self.assertEqual(limited, subjects[:3])

    def test_limit_subject_dirs_keeps_all_subjects_when_limit_is_none(self):
        subjects = [Path(f"subject-{i}") for i in range(5)]

        limited = limit_subject_dirs(subjects, None, "fold 1")

        self.assertEqual(limited, subjects)

    def test_limit_subject_dirs_rejects_non_positive_limit(self):
        with self.assertRaises(ValueError):
            limit_subject_dirs([Path("subject-0")], 0, "fold 1")


if __name__ == "__main__":
    unittest.main()
