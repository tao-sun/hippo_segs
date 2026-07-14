import unittest

from torch.utils.data import Dataset

from snn_fptt import make_overfit_datasets


class DummySubjectDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.view = "axial"

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return f"subject-{idx}"


class OverfitSubsetTest(unittest.TestCase):
    def test_make_overfit_datasets_uses_same_subjects_for_train_and_val(self):
        train_ds = DummySubjectDataset(5)

        overfit_train, overfit_val = make_overfit_datasets(train_ds, "axial", 3)

        self.assertEqual(len(overfit_train), 3)
        self.assertEqual(len(overfit_val), 3)
        self.assertEqual(overfit_train.view, "axial")
        self.assertEqual(overfit_val.view, "axial")
        self.assertEqual([overfit_train[i] for i in range(3)], ["subject-0", "subject-1", "subject-2"])
        self.assertEqual([overfit_val[i] for i in range(3)], ["subject-0", "subject-1", "subject-2"])

    def test_make_overfit_datasets_rejects_non_positive_subject_count(self):
        with self.assertRaises(ValueError):
            make_overfit_datasets(DummySubjectDataset(5), "axial", 0)


if __name__ == "__main__":
    unittest.main()
