"""Tests for the specialist referral feature.

The first group pins down the requirements as the project review set them:
five hospitals, five doctors each, and suggestions only for grades 2-4. The
second group checks the ranking actually does what the module claims.

Run:  python -m unittest discover -s tests -v
"""
import json
import tempfile
import unittest
from pathlib import Path

from src import config as C
from src import referral as R


class DirectoryMeetsReviewSpec(unittest.TestCase):
    """What the review asked for: 5 hospitals x 5 doctors, numbers included."""

    def setUp(self):
        self.hospitals = R.load_directory()

    def test_five_hospitals(self):
        self.assertEqual(len(self.hospitals), 5)

    def test_five_doctors_per_hospital(self):
        for h in self.hospitals:
            with self.subTest(hospital=h.name):
                self.assertEqual(len(h.doctors), 5)

    def test_every_doctor_has_a_name_and_number(self):
        for h in self.hospitals:
            for d in h.doctors:
                with self.subTest(doctor=d.name):
                    self.assertTrue(d.name.startswith("Dr. "))
                    self.assertRegex(d.phone, R.FICTIONAL_PHONE)

    def test_directory_is_declared_fictional(self):
        raw = json.loads(C.HOSPITAL_DIRECTORY.read_text(encoding="utf-8"))
        self.assertIs(raw["meta"]["fictional"], True)

    def test_all_numbers_unique_and_in_fictional_range(self):
        phones = [h.phone for h in self.hospitals] + [
            d.phone for h in self.hospitals for d in h.doctors]
        self.assertEqual(len(phones), len(set(phones)))
        for p in phones:
            self.assertRegex(p, R.FICTIONAL_PHONE)


class SuggestionsOnlyForReferableGrades(unittest.TestCase):
    """The review's rule: suggest for grades 2, 3, 4 - never for 0 or 1."""

    def test_no_suggestion_for_grades_0_and_1(self):
        for grade in (0, 1):
            with self.subTest(grade=grade):
                self.assertIsNone(R.recommend(grade))

    def test_suggestion_for_grades_2_to_4(self):
        for grade in (2, 3, 4):
            with self.subTest(grade=grade):
                ref = R.recommend(grade)
                self.assertIsNotNone(ref)
                self.assertIsNotNone(ref.best)

    def test_threshold_is_the_shared_referable_grade(self):
        # The referral line must be the same one the prediction output and
        # the evaluation metrics use, or the app could say "Referable" and
        # then show no hospitals.
        self.assertEqual(C.REFERABLE_GRADE, 2)
        self.assertIsNone(R.recommend(C.REFERABLE_GRADE - 1))
        self.assertIsNotNone(R.recommend(C.REFERABLE_GRADE))

    def test_invalid_grade_rejected(self):
        for grade in (-1, 5, 10):
            with self.subTest(grade=grade), self.assertRaises(ValueError):
                R.recommend(grade)


class RankingIsSeverityMatched(unittest.TestCase):

    def test_best_hospital_differs_by_grade(self):
        best = {g: R.recommend(g).best.hospital.id for g in (2, 3, 4)}
        self.assertEqual(best, {2: "anvaya", 3: "vaidurya", 4: "kaustubha"})

    def test_every_suggestion_offers_the_required_services(self):
        for grade in (2, 3, 4):
            ref = R.recommend(grade)
            for m in ref.matches:
                with self.subTest(grade=grade, hospital=m.hospital.name):
                    self.assertLessEqual(ref.rule.required,
                                         m.hospital.capabilities)

    def test_hospitals_without_surgery_never_suggested_for_grade_4(self):
        for m in R.recommend(4).matches:
            self.assertIn("vitrectomy", m.hospital.capabilities)

    def test_listed_doctors_suit_the_grade(self):
        for grade in (2, 3, 4):
            ref = R.recommend(grade)
            for m in ref.matches:
                for d in m.doctors:
                    with self.subTest(grade=grade, doctor=d.name):
                        self.assertIn(d.specialty, ref.rule.specialties)

    def test_primary_specialists_listed_first(self):
        for grade in (2, 3, 4):
            ref = R.recommend(grade)
            for m in ref.matches:
                specs = [d.specialty for d in m.doctors]
                order = [ref.rule.specialties.index(s) for s in specs]
                with self.subTest(grade=grade, hospital=m.hospital.name):
                    self.assertEqual(order, sorted(order))

    def test_matches_ordered_by_score(self):
        for grade in (2, 3, 4):
            scores = [m.score for m in R.recommend(grade).matches]
            with self.subTest(grade=grade):
                self.assertEqual(scores, sorted(scores, reverse=True))

    def test_ranking_is_deterministic(self):
        for grade in (2, 3, 4):
            a = [m.hospital.id for m in R.recommend(grade).matches]
            b = [m.hospital.id for m in R.recommend(grade).matches]
            self.assertEqual(a, b)


class ValidationCatchesBadData(unittest.TestCase):

    def _write(self, mutate):
        raw = json.loads(C.HOSPITAL_DIRECTORY.read_text(encoding="utf-8"))
        mutate(raw)
        tmp = Path(tempfile.mkdtemp()) / "directory.json"
        tmp.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        return tmp

    def test_rejects_a_real_looking_number(self):
        def m(raw): raw["hospitals"][0]["doctors"][0]["phone"] = "+91 98450 12345"
        with self.assertRaises(R.DirectoryError):
            R.load_directory(self._write(m))

    def test_rejects_unknown_specialty(self):
        def m(raw): raw["hospitals"][0]["doctors"][0]["specialty"] = "Dentist"
        with self.assertRaises(R.DirectoryError):
            R.load_directory(self._write(m))

    def test_rejects_unknown_capability(self):
        def m(raw): raw["hospitals"][0]["capabilities"].append("teleportation")
        with self.assertRaises(R.DirectoryError):
            R.load_directory(self._write(m))

    def test_rejects_duplicate_ids(self):
        def m(raw): raw["hospitals"][1]["id"] = raw["hospitals"][0]["id"]
        with self.assertRaises(R.DirectoryError):
            R.load_directory(self._write(m))

    def test_rejects_missing_field(self):
        def m(raw): del raw["hospitals"][0]["phone"]
        with self.assertRaises(R.DirectoryError):
            R.load_directory(self._write(m))


if __name__ == "__main__":
    unittest.main()
