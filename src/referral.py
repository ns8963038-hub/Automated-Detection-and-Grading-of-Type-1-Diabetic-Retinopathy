"""Specialist referral suggestions for referable diabetic retinopathy.

When an image is graded 2, 3 or 4 - the referable threshold - the app
suggests where the patient should go next: the best-matched hospital from a
small directory, together with the doctors there who suit that grade. Grades
0 and 1 get no suggestion, because they call for routine re-screening rather
than a specialist.

"Best" is defined by severity, not by size. Each grade carries

  required services   a hospital lacking any of them is not suggested at all.
                      Proliferative DR may need laser and vitreoretinal
                      surgery, so a clinic offering neither is no use however
                      highly it scores on anything else.
  preferred services  each adds points - 24x7 emergency care matters for
                      grade 4, a diabetic eye clinic suits grade 2.
  primary specialty   one point per doctor of that kind on staff, so a
                      centre with three retina specialists beats one with a
                      single specialist for a retina problem.

The highest total wins. The rule is deliberately simple so a suggestion can
always be explained: every point traces back to a named service or a doctor.

The directory lives in data/hospital_directory.json. Every hospital, doctor
and phone number in it is fictional.

Run:  python -m src.referral --grade 4
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src import config as C

# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------
CAPABILITIES = {                       # display order follows this order
    "ophthalmology": "General eye care",
    "diabetic_eye_clinic": "Diabetic eye clinic",
    "medical_retina": "Medical retina clinic",
    "laser": "Retinal laser treatment",
    "anti_vegf": "Anti-VEGF injections",
    "vitrectomy": "Vitreoretinal surgery",
    "emergency_24x7": "24×7 emergency eye care",
}

SPECIALTIES = {                        # singular, plural
    "Vitreoretinal Surgeon": ("vitreoretinal surgeon", "vitreoretinal surgeons"),
    "Medical Retina Specialist": ("medical retina specialist",
                                  "medical retina specialists"),
    "General Ophthalmologist": ("general ophthalmologist",
                                "general ophthalmologists"),
}

# Every number in the sample directory sits in +91 80 0XXX XXXX. Indian
# subscriber numbers do not begin with 0 (it is the trunk prefix), so these
# should not reach a real line. Validation enforces the range, which stops a
# plausible-looking real number being typed into the file by accident.
FICTIONAL_PHONE = re.compile(r"^\+91 80 0\d{3} \d{4}$")


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Doctor:
    name: str
    specialty: str
    qualifications: str
    experience_years: int
    availability: str
    phone: str


@dataclass(frozen=True)
class Hospital:
    id: str
    name: str
    kind: str
    area: str
    city: str
    phone: str
    hours: str
    capabilities: frozenset[str]
    doctors: tuple[Doctor, ...]

    @property
    def location(self) -> str:
        return f"{self.area}, {self.city}"


@dataclass(frozen=True)
class Rule:
    urgency: str
    advice: str
    required: frozenset[str]
    preferred: tuple[tuple[str, int], ...]   # (service, points), display order
    specialties: tuple[str, ...]             # suitable specialties, best first

    @property
    def primary(self) -> str:
        return self.specialties[0]


@dataclass(frozen=True)
class Match:
    hospital: Hospital
    score: int
    doctors: tuple[Doctor, ...]              # suited to the grade, best first
    reasons: tuple[str, ...]                 # what earned the score


@dataclass(frozen=True)
class Referral:
    grade: int
    rule: Rule
    matches: tuple[Match, ...]               # best first

    @property
    def best(self) -> Match | None:
        return self.matches[0] if self.matches else None

    @property
    def others(self) -> tuple[Match, ...]:
        return self.matches[1:]


# --------------------------------------------------------------------------
# Referral rules, one per referable grade
# --------------------------------------------------------------------------
RULES = {
    2: Rule(
        urgency="Routine referral",
        advice="Book a review with an ophthalmologist, who will decide how "
               "closely the eye needs monitoring. A clinic set up for "
               "diabetic eye follow-up is the best fit at this stage.",
        required=frozenset({"ophthalmology"}),
        preferred=(("diabetic_eye_clinic", 3), ("medical_retina", 1)),
        specialties=("Medical Retina Specialist", "General Ophthalmologist"),
    ),
    3: Rule(
        urgency="Prompt referral",
        advice="See a retina specialist soon. Severe NPDR carries a high risk "
               "of progressing to proliferative disease and may need laser "
               "treatment, so only hospitals offering it are suggested.",
        required=frozenset({"medical_retina", "laser"}),
        preferred=(("anti_vegf", 2), ("diabetic_eye_clinic", 1)),
        specialties=("Medical Retina Specialist", "Vitreoretinal Surgeon"),
    ),
    4: Rule(
        urgency="Urgent referral",
        advice="See a vitreoretinal specialist as soon as possible. "
               "Proliferative DR may need laser, injections or surgery to "
               "protect vision, so only hospitals able to operate are "
               "suggested.",
        required=frozenset({"laser", "vitrectomy"}),
        preferred=(("emergency_24x7", 3), ("anti_vegf", 2)),
        specialties=("Vitreoretinal Surgeon", "Medical Retina Specialist"),
    ),
}


# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------
class DirectoryError(ValueError):
    """The hospital directory file is missing fields or inconsistent."""


def _parse_doctor(raw: dict, where: str) -> Doctor:
    try:
        doc = Doctor(
            name=str(raw["name"]),
            specialty=str(raw["specialty"]),
            qualifications=str(raw["qualifications"]),
            experience_years=int(raw["experience_years"]),
            availability=str(raw["availability"]),
            phone=str(raw["phone"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DirectoryError(f"{where}: malformed doctor entry ({exc})") from exc
    if doc.specialty not in SPECIALTIES:
        raise DirectoryError(f"{where}: unknown specialty {doc.specialty!r}")
    if doc.experience_years < 0:
        raise DirectoryError(f"{where}: negative experience for {doc.name}")
    return doc


def _parse_hospital(raw: dict) -> Hospital:
    where = raw.get("id", raw.get("name", "<unnamed hospital>"))
    try:
        caps = frozenset(raw["capabilities"])
        doctors = tuple(_parse_doctor(d, where) for d in raw["doctors"])
        hosp = Hospital(
            id=str(raw["id"]), name=str(raw["name"]), kind=str(raw["type"]),
            area=str(raw["area"]), city=str(raw["city"]),
            phone=str(raw["phone"]), hours=str(raw["hours"]),
            capabilities=caps, doctors=doctors,
        )
    except (KeyError, TypeError) as exc:
        raise DirectoryError(f"{where}: missing field {exc}") from exc
    unknown = caps - CAPABILITIES.keys()
    if unknown:
        raise DirectoryError(f"{where}: unknown capabilities {sorted(unknown)}")
    if not doctors:
        raise DirectoryError(f"{where}: no doctors listed")
    return hosp


def _validate(hospitals: tuple[Hospital, ...]) -> None:
    if not hospitals:
        raise DirectoryError("directory lists no hospitals")
    ids = [h.id for h in hospitals]
    if len(ids) != len(set(ids)):
        raise DirectoryError("hospital ids are not unique")
    phones = [h.phone for h in hospitals] + [
        d.phone for h in hospitals for d in h.doctors]
    if len(phones) != len(set(phones)):
        raise DirectoryError("phone numbers are not unique")
    real_looking = [p for p in phones if not FICTIONAL_PHONE.match(p)]
    if real_looking:
        raise DirectoryError(
            "numbers outside the fictional +91 80 0XXX XXXX range: "
            + ", ".join(real_looking))


@lru_cache(maxsize=4)
def _load(path: str, mtime_ns: int) -> tuple[Hospital, ...]:
    # Explicit UTF-8: on Windows the default encoding is cp1252, which would
    # mangle the en dashes and the multiplication sign in this file.
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        entries = raw["hospitals"]
    except (KeyError, TypeError) as exc:
        raise DirectoryError("top-level 'hospitals' list missing") from exc
    hospitals = tuple(_parse_hospital(h) for h in entries)
    _validate(hospitals)
    return hospitals


def load_directory(path: str | Path | None = None) -> tuple[Hospital, ...]:
    """Load and validate the directory.

    Cached on the file's modification time, so editing the JSON while the app
    is running takes effect on the next prediction without a restart.
    """
    path = Path(path or C.HOSPITAL_DIRECTORY)
    return _load(str(path), path.stat().st_mtime_ns)


# --------------------------------------------------------------------------
# Recommendation
# --------------------------------------------------------------------------
def _count(n: int, specialty: str) -> str:
    singular, plural = SPECIALTIES[specialty]
    return f"{n} {singular if n == 1 else plural}"


def _match(hospital: Hospital, rule: Rule) -> Match | None:
    if not rule.required <= hospital.capabilities:
        return None
    rank = {s: i for i, s in enumerate(rule.specialties)}
    suited = tuple(sorted(
        (d for d in hospital.doctors if d.specialty in rank),
        key=lambda d: (rank[d.specialty], -d.experience_years, d.name),
    ))
    if not suited:                     # the services exist but nobody to see
        return None

    primary = sum(1 for d in suited if d.specialty == rule.primary)
    services = sum(pts for cap, pts in rule.preferred
                   if cap in hospital.capabilities)

    # Reasons in a fixed order: what makes the hospital eligible, then what
    # earned extra points, then the specialists. "General eye care" is left
    # out because every eye hospital offers it and it explains nothing.
    eligible = [CAPABILITIES[c] for c in CAPABILITIES
                if c in rule.required and c != "ophthalmology"]
    extras = [CAPABILITIES[c] for c, _ in rule.preferred
              if c in hospital.capabilities]
    reasons = tuple(eligible + extras + [_count(primary, rule.primary)])
    return Match(hospital, services + primary, suited, reasons)


def recommend(grade: int,
              hospitals: tuple[Hospital, ...] | None = None) -> Referral | None:
    """Rank the directory for a predicted grade.

    Returns None below the referable threshold - grades 0 and 1 get no
    suggestion. Ties are broken by depth of the primary specialty, then by
    the most experienced suitable doctor, then by name, so the order is
    always deterministic.
    """
    if grade not in range(len(C.CLASS_NAMES)):
        raise ValueError(f"grade must be 0-{len(C.CLASS_NAMES) - 1}, got {grade}")
    if grade < C.REFERABLE_GRADE:
        return None

    rule = RULES[grade]
    hospitals = hospitals if hospitals is not None else load_directory()
    matches = [m for m in (_match(h, rule) for h in hospitals) if m]

    def order(m: Match):
        primary = sum(1 for d in m.doctors if d.specialty == rule.primary)
        return (-m.score, -primary, -m.doctors[0].experience_years,
                m.hospital.name)

    return Referral(grade, rule, tuple(sorted(matches, key=order)))


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Show specialist suggestions for a DR grade.")
    ap.add_argument("--grade", type=int, required=True, choices=range(5))
    args = ap.parse_args()

    ref = recommend(args.grade)
    print(f"\nGrade {C.CLASS_NAMES[args.grade]}")
    if ref is None:
        print("Below the referral threshold: routine screening, "
              "no specialist suggested.\n")
        return
    print(f"{ref.rule.urgency}. {ref.rule.advice}\n")
    for i, m in enumerate(ref.matches, 1):
        tag = "BEST MATCH" if i == 1 else f"#{i}"
        print(f"  {tag:<11} {m.hospital.name}  ({m.hospital.location})  "
              f"score {m.score}")
        print(f"              {' · '.join(m.reasons)}")
        for d in m.doctors:
            print(f"              - {d.name:<22} {d.specialty:<26} "
                  f"{d.experience_years:>2} yrs  {d.phone}")
        print()


if __name__ == "__main__":
    main()
