"""Synthetic stand-in for the challenge data, in the exact file format.

It exists so every stage can be built and tested before the real data
arrives. The noise follows the problem statement: legal-suffix variants,
abbreviations, "&"/"and", word-order swaps, typos, transliteration variants,
DBA names, missing PIN/state, landmark references, reordered components. It
also plants the traps that make the metric hard: chain branches that share a
name at different addresses, near-identical names of different businesses,
Source 2/3 records with no Source 1 counterpart, and singletons. France appears
only in the test split, as in the real challenge.

Scores on synthetic data say the pipeline works, not how it will rank.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from ber import schema
from ber.io import write_id_lists

# fmt: off
US_FIRST = ["Golden", "Silver", "Blue", "Green", "Eagle", "Pioneer", "Summit", "Liberty", "Pacific", "Atlantic",
            "Northern", "Southern", "Great", "Royal", "Prime", "Apex", "Bright", "Sunrise", "Evergreen", "Maple",
            "Cedar", "River", "Lakeside", "Mountain", "Valley", "Coastal", "Metro", "Urban", "First", "United",
            "American", "National", "Global", "Advanced", "Precision", "Elite", "Superior", "Classic", "Heritage", "Keystone"]
US_SECOND = ["Gate", "Star", "Bridge", "Point", "Ridge", "Harbor", "Crest", "Field", "Stone", "Wood", "Line", "View",
             "Way", "Path", "Peak", "Rock", "Brook", "Hill", "Grove", "Park"]
US_TYPE = ["Plumbing", "Dental", "Bakery", "Auto Repair", "Consulting", "Logistics", "Pharmacy", "Hardware", "Realty",
           "Insurance", "Construction", "Electric", "Software", "Cafe", "Pizza", "Fitness", "Cleaning Services",
           "Landscaping", "Law Group", "Tax & Accounting", "Printing", "Furniture", "Flooring", "Roofing",
           "Veterinary Clinic", "Market", "Motors", "Travel", "Marketing", "Technologies", "Heating & Cooling",
           "Brothers Moving", "International Trading"]
US_SUFFIX = ["LLC", "Inc", "Corp", "Corporation", "Co", "Company", "Ltd", "", ""]
US_STREETS = ["Main", "Oak", "Maple", "Washington", "Lake", "Hill", "Park", "Pine", "Cedar", "Elm", "Sunset", "Highland",
              "Lincoln", "Jefferson", "Madison", "Franklin", "Church", "Market", "Spring", "River", "Mill", "Broad"]
US_STYPES = [("Street", "St"), ("Avenue", "Ave"), ("Road", "Rd"), ("Boulevard", "Blvd"), ("Drive", "Dr"),
             ("Lane", "Ln"), ("Court", "Ct"), ("Parkway", "Pkwy")]
US_CITIES = [("Austin", "TX", "Texas", "787"), ("Seattle", "WA", "Washington", "981"), ("Denver", "CO", "Colorado", "802"),
             ("Boston", "MA", "Massachusetts", "021"), ("Chicago", "IL", "Illinois", "606"),
             ("Phoenix", "AZ", "Arizona", "850"), ("Portland", "OR", "Oregon", "972"), ("Atlanta", "GA", "Georgia", "303"),
             ("Miami", "FL", "Florida", "331"), ("San Jose", "CA", "California", "951"),
             ("Columbus", "OH", "Ohio", "432"), ("Nashville", "TN", "Tennessee", "372")]

IN_FIRST = ["Sri Lakshmi", "Shree Ganesh", "Balaji", "Sai", "Krishna", "Durga", "Annapurna", "Venkateswara",
            "Mahalakshmi", "Hanuman", "Om Sai", "Jai Hind", "Bharat", "Surya", "Ganga", "Kaveri", "Anand", "Vijay",
            "Murugan", "Ambika", "Saraswati", "Navrang", "Royal", "New India", "Shiv Shakti"]
IN_MIDDLE = ["Ramesh", "Suresh", "Mahesh", "Kumar", "Reddy", "Sharma", "Patel", "Gupta", "Iyer", "Nair", "Rao",
             "Agarwal", "Jain", "Mehta", "Singh", "Das", "Bose", "Pillai", "Menon", "Chettiar"]
IN_TYPE = ["Traders", "Enterprises", "Textiles", "Jewellers", "Medicals", "Stores", "Electronics", "Hardwares",
           "Sweets", "Agencies", "Motors", "Industries", "Foods", "Constructions", "Pharma", "Steels",
           "Tours and Travels", "Hotel", "Silks", "Mobiles", "Opticals", "Furnitures"]
IN_SUFFIX = ["Pvt Ltd", "Private Limited", "Pvt. Ltd.", "Ltd", "Limited", "LLP", "", "", ""]
IN_STREETS = ["MG", "Station", "Gandhi", "Nehru", "Temple", "Market", "Church", "Hosur", "Mount", "Link", "Ring",
              "Lake", "Subhash", "Tilak", "Rajaji", "Old Airport"]
IN_STYPES = [("Road", "Rd"), ("Main Road", "Main Rd"), ("Street", "St"), ("Cross Road", "Cross Rd"), ("Marg", "Marg")]
IN_AREAS = ["Jayanagar", "Koramangala", "Indiranagar", "T Nagar", "Anna Nagar", "Andheri West", "Bandra", "Karol Bagh",
            "Salt Lake", "Banjara Hills", "Kothrud", "Malleswaram", "Adyar", "Lajpat Nagar", "Viman Nagar"]
IN_CITIES = [("Bengaluru", "Bangalore", "Karnataka", "560"), ("Chennai", "Madras", "Tamil Nadu", "600"),
             ("Mumbai", "Bombay", "Maharashtra", "400"), ("New Delhi", "Delhi", "Delhi", "110"),
             ("Kolkata", "Calcutta", "West Bengal", "700"), ("Hyderabad", "Hyderabad", "Telangana", "500"),
             ("Pune", "Poona", "Maharashtra", "411")]
IN_LANDMARKS = ["Near SBI ATM", "Opp. Bus Stand", "Behind Railway Station", "Near Hanuman Temple",
                "Opposite City Hospital", "Next to HDFC Bank", "Near Post Office", "Beside Big Bazaar"]
TRANSLIT = {"Lakshmi": ["Laxmi", "Lakshmee"], "Shree": ["Shri", "Sri", "Sree"], "Sri": ["Shri", "Shree", "Sree"],
            "Krishna": ["Krushna", "Krisna"], "Venkateswara": ["Venkateshwara", "Venkatesvara"],
            "Jewellers": ["Jewelers"], "Balaji": ["Baalaji"], "Saraswati": ["Saraswathi"], "Murugan": ["Muruga"],
            "Mahalakshmi": ["Mahalaxmi"], "Surya": ["Suriya"], "Kaveri": ["Cauvery"], "Ganesh": ["Ganesha"],
            "Shiv": ["Shiva"], "Chettiar": ["Chettiyar"], "Iyer": ["Iyyer"], "Agarwal": ["Aggarwal"]}

FR_FIRST = ["Boulangerie", "Pharmacie", "Garage", "Cabinet", "Librairie", "Fromagerie", "Brasserie", "Atelier",
            "Menuiserie", "Plomberie", "Transports", "Boucherie", "Hôtel", "Café", "Société", "Pâtisserie", "Optique"]
FR_NAMES = ["Dupont", "Martin", "Bernard", "Lefèvre", "Moreau", "Laurent", "Girard", "Rousseau", "Fournier", "Mercier",
            "Bonnet", "Lambert", "Fontaine", "Chevalier", "Sainte-Marie", "Saint-Michel", "du Centre", "de la Gare",
            "des Arts", "Beaulieu", "Rivière", "Mérieux"]
FR_SUFFIX = ["SARL", "SAS", "SA", "EURL", "& Fils", "et Fils", "", ""]
FR_STYPES = [("rue", "r."), ("avenue", "av."), ("boulevard", "bd"), ("place", "pl."), ("chemin", "ch."),
             ("allée", "all."), ("impasse", "imp.")]
FR_STREETS = ["de la République", "Victor Hugo", "Jean Jaurès", "de Paris", "Pasteur", "du Général de Gaulle",
              "des Écoles", "Saint-Honoré", "de la Paix", "Gambetta", "Émile Zola", "de la Liberté"]
FR_CITIES = [("Paris", "750"), ("Lyon", "690"), ("Marseille", "130"), ("Toulouse", "310"), ("Nice", "060"),
             ("Nantes", "440"), ("Strasbourg", "670"), ("Bordeaux", "330"), ("Lille", "590"), ("Rennes", "350")]

SUFFIX_VARIANTS = {"LLC": ["L.L.C.", "llc", "Limited Liability Company", ""], "Inc": ["Inc.", "Incorporated", ""],
                   "Corp": ["Corp.", "Corporation", ""], "Corporation": ["Corp", "Corp.", ""], "Co": ["Co.", "Company", ""],
                   "Company": ["Co", "Co.", ""], "Ltd": ["Ltd.", "Limited", ""], "Limited": ["Ltd", "Ltd.", ""],
                   "Pvt Ltd": ["Pvt. Ltd.", "Private Limited", "P Ltd", ""], "Private Limited": ["Pvt Ltd", "Pvt. Ltd.", ""],
                   "Pvt. Ltd.": ["Pvt Ltd", "Private Limited", ""], "LLP": ["L.L.P.", ""], "SARL": ["S.A.R.L.", "Sarl", ""],
                   "SAS": ["S.A.S.", ""], "SA": ["S.A.", ""], "EURL": ["Eurl", ""], "& Fils": ["et Fils", "and Sons"],
                   "et Fils": ["& Fils"]}
WORD_ABBREV = {"Services": "Svcs", "International": "Intl", "Technologies": "Tech", "Brothers": "Bros",
               "Construction": "Constr", "Consulting": "Cnsltg", "Enterprises": "Ent", "Industries": "Inds",
               "Company": "Co", "Travel": "Travels", "Cleaning": "Clean"}
# fmt: on


@dataclass
class Entity:
    country: str
    words: list[str]
    suffix: str
    num: str
    street: str
    stype: tuple[str, str]
    city: str
    city_alt: str
    state: str
    state_full: str
    postal: str
    area: str = ""
    unit: str = ""
    landmark: str = ""


@dataclass
class SynthConfig:
    train_entities: int = 3000
    test_entities: int = 2000
    seed: int = 7
    s1_fraction: float = 0.8  # entities present in Source 1; the rest exist only in Source 2/3
    branch_fraction: float = 0.08  # entities that reuse an existing name at another address
    colocated_fraction: float = (
        0.08  # different businesses at the same address (mall, office block)
    )
    sparse_address_rate: float = 0.06  # noisy copies whose address is little more than a city
    alias_only_rate: float = 0.02  # noisy copies listed under an unrelated trade name
    copy_weights: tuple[float, ...] = (
        0.40,
        0.42,
        0.13,
        0.05,
    )  # P(0, 1, 2, 3 copies) in each of S2 and S3
    train_countries: tuple[str, ...] = ("US", "India")
    test_countries: tuple[str, ...] = ("US", "India", "France")
    notes: list[str] = field(default_factory=list)


class _Generator:
    def __init__(self, rng: random.Random, cfg: SynthConfig) -> None:
        self.r = rng
        self.cfg = cfg

    def p(self, prob: float) -> bool:
        return self.r.random() < prob

    # ---- entities
    def entity(self, country: str) -> Entity:
        r = self.r
        if country == "US":
            city, state, state_full, zip3 = r.choice(US_CITIES)
            words = [r.choice(US_FIRST), r.choice(US_SECOND), *r.choice(US_TYPE).split()]
            if self.p(0.3):
                words = [r.choice(US_FIRST), *r.choice(US_TYPE).split()]
            return Entity(
                country,
                words,
                r.choice(US_SUFFIX),
                str(r.randint(1, 9999)),
                r.choice(US_STREETS),
                r.choice(US_STYPES),
                city,
                city,
                state,
                state_full,
                f"{zip3}{r.randint(0, 99):02d}",
                unit=f"Suite {r.randint(100, 999)}" if self.p(0.2) else "",
            )
        if country == "India":
            city, alt, state, pin3 = r.choice(IN_CITIES)
            words = [*r.choice(IN_FIRST).split()]
            if self.p(0.5):
                words.append(r.choice(IN_MIDDLE))
            words += r.choice(IN_TYPE).split()
            return Entity(
                country,
                words,
                r.choice(IN_SUFFIX),
                str(r.randint(1, 450)),
                r.choice(IN_STREETS),
                r.choice(IN_STYPES),
                city,
                alt,
                state,
                state,
                f"{pin3}{r.randint(0, 99):03d}",
                area=r.choice(IN_AREAS),
                landmark=r.choice(IN_LANDMARKS) if self.p(0.4) else "",
            )
        if country == "France":
            city, cp3 = r.choice(FR_CITIES)
            words = [r.choice(FR_FIRST), *r.choice(FR_NAMES).split()]
            return Entity(
                country,
                words,
                r.choice(FR_SUFFIX),
                str(r.randint(1, 180)),
                r.choice(FR_STREETS),
                r.choice(FR_STYPES),
                city,
                city,
                "",
                "",
                f"{cp3}{r.randint(0, 20):02d}",
            )
        raise ValueError(f"no generator for country {country!r}")

    def branch(self, base: Entity) -> Entity:
        """Same business name, different location (often the same city): a different entity."""
        other = self.entity(base.country)
        other.words, other.suffix = list(base.words), base.suffix
        if self.p(0.5):
            other.city, other.city_alt, other.state, other.state_full = (
                base.city,
                base.city_alt,
                base.state,
                base.state_full,
            )
            other.postal = base.postal if self.p(0.5) else other.postal
        return other

    def colocated(self, base: Entity) -> Entity:
        """A different business at the same street address."""
        other = self.entity(base.country)
        for attr in (
            "num",
            "street",
            "stype",
            "city",
            "city_alt",
            "state",
            "state_full",
            "postal",
            "area",
            "landmark",
        ):
            setattr(other, attr, getattr(base, attr))
        return other

    # ---- rendering
    def name(self, e: Entity, noise: float) -> str:
        r = self.r
        words = list(e.words)
        suffix = e.suffix
        if self.p(noise * 0.6) and suffix in SUFFIX_VARIANTS:
            suffix = r.choice(SUFFIX_VARIANTS[suffix])
        if self.p(noise * 0.4):
            words = [r.choice(TRANSLIT[w]) if w in TRANSLIT and self.p(0.7) else w for w in words]
        if self.p(noise * 0.2):
            words = [WORD_ABBREV.get(w, w) for w in words]
        if self.p(noise * 0.3):
            words = [{"&": "and", "and": "&", "et": "&"}.get(w, w) for w in words]
        if self.p(noise * 0.12) and len(words) >= 2:
            i = r.randrange(len(words) - 1)
            words[i], words[i + 1] = words[i + 1], words[i]
        if self.p(noise * 0.25):
            words = self._typo(words)
        if self.p(noise * 0.08):
            words = self._typo(words)
        text = " ".join([*words, suffix]).strip()
        if self.p(noise * self.cfg.alias_only_rate):
            return self._case(f"{r.choice(US_FIRST)} {r.choice(US_SECOND)} {words[-1]}", noise)
        if self.p(noise * 0.04):
            text = f"{text} DBA {r.choice(US_FIRST)} {words[-1]}"
        if e.country == "France" and self.p(noise * 0.4):
            text = _strip_accents(text)
        return self._case(text, noise)

    def address(self, e: Entity, noise: float) -> str:
        r = self.r
        full, short = e.stype
        stype = short if self.p(0.5) else full
        street = e.street if not self.p(noise * 0.15) else self._typo([e.street])[0]
        postal = "" if self.p(noise * 0.25) else e.postal
        if self.p(noise * self.cfg.sparse_address_rate):
            return self._case(
                ", ".join(p for p in (e.city if self.p(0.5) else e.city_alt, postal) if p), noise
            )
        if e.country == "US":
            if postal and self.p(noise * 0.15):
                postal = f"{postal}-{r.randint(1000, 9999)}"
            state = "" if self.p(noise * 0.2) else (e.state_full if self.p(0.2) else e.state)
            parts = [
                f"{e.num} {street} {stype}",
                e.unit if not self.p(noise * 0.4) else "",
                e.city,
                f"{state} {postal}".strip(),
            ]
            if self.p(noise * 0.08):
                parts = [parts[2], parts[3], parts[0], parts[1]]
        elif e.country == "India":
            num = (
                r.choice([f"No. {e.num}", f"#{e.num}", e.num, f"{e.num}/{r.randint(1, 9)}"])
                if self.p(noise)
                else f"No. {e.num}"
            )
            city = e.city_alt if self.p(noise * 0.4) else e.city
            if postal and self.p(noise * 0.2):
                postal = f"{postal[:3]} {postal[3:]}"
            landmark = (
                e.landmark
                if self.p(0.6)
                else (r.choice(IN_LANDMARKS) if self.p(noise * 0.3) else "")
            )
            state = "" if self.p(noise * 0.35) else e.state
            area = "" if self.p(noise * 0.2) else e.area
            tail = (
                f"{city} - {postal}"
                if postal and self.p(0.5)
                else f"{city}, {state} {postal}".strip(" ,")
            )
            parts = [f"{num}, {street} {stype}", landmark, area, tail]
            if self.p(noise * 0.15):
                parts = [landmark, parts[0], area, tail]
        else:
            cp_city = f"{postal} {e.city}".strip()
            parts = [f"{e.num} {stype} {street}", cp_city]
            if self.p(noise * 0.1):
                parts = [f"{e.num}, {stype} {street}", e.city, postal]
        text = ", ".join(p for p in parts if p)
        if e.country == "France" and self.p(noise * 0.4):
            text = _strip_accents(text)
        return self._case(text, noise * 0.5)

    def _typo(self, words: list[str]) -> list[str]:
        r = self.r
        candidates = [i for i, w in enumerate(words) if len(w) > 3]
        if not candidates:
            return words
        i = r.choice(candidates)
        w = words[i]
        j = r.randrange(1, len(w) - 1)
        op = r.randrange(3)
        if op == 0:
            w = w[:j] + w[j + 1 :]
        elif op == 1:
            w = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
        else:
            w = w[:j] + r.choice("aeiourstn") + w[j + 1 :]
        return [*words[:i], w, *words[i + 1 :]]

    def _case(self, text: str, noise: float) -> str:
        roll = self.r.random()
        if roll < noise * 0.15:
            return text.upper()
        if roll < noise * 0.25:
            return text.lower()
        return text


def _strip_accents(text: str) -> str:
    import unicodedata

    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def generate_split(
    n_entities: int, countries: tuple[str, ...], cfg: SynthConfig, rng: random.Random
) -> tuple[list[list[dict]], dict[str, list[str]]]:
    gen = _Generator(rng, cfg)
    entities: list[Entity] = []
    for _ in range(n_entities):
        country = rng.choice(countries)
        same_country = [e for e in entities[-200:] if e.country == country]
        roll = rng.random()
        if same_country and roll < cfg.branch_fraction:
            entities.append(gen.branch(rng.choice(same_country)))
        elif same_country and roll < cfg.branch_fraction + cfg.colocated_fraction:
            entities.append(gen.colocated(rng.choice(same_country)))
        else:
            entities.append(gen.entity(country))

    records: dict[int, list[tuple[int, dict]]] = {1: [], 2: [], 3: []}
    for k, e in enumerate(entities):
        in_s1 = gen.p(cfg.s1_fraction)
        if in_s1:
            records[1].append(
                (
                    k,
                    {
                        "name": gen.name(e, 0.25),
                        "address": gen.address(e, 0.25),
                        "country": e.country,
                    },
                )
            )
        for source in (2, 3):
            copies = rng.choices(range(len(cfg.copy_weights)), weights=cfg.copy_weights)[0]
            for _ in range(copies):
                records[source].append(
                    (
                        k,
                        {
                            "name": gen.name(e, 1.0),
                            "address": gen.address(e, 1.0),
                            "country": e.country,
                        },
                    )
                )

    rows: list[list[dict]] = []
    entity_to_ids: dict[int, dict[int, list[str]]] = {1: {}, 2: {}, 3: {}}
    for source in (1, 2, 3):
        items = records[source]
        rng.shuffle(items)
        width = max(5, len(str(len(items))))
        source_rows = []
        for i, (k, rec) in enumerate(items, start=1):
            eid = f"{schema.SOURCE_PREFIX[source]}{i:0{width}d}"
            entity_to_ids[source].setdefault(k, []).append(eid)
            source_rows.append(
                {
                    schema.ENTITY_ID: eid,
                    schema.NAME: rec["name"],
                    schema.ADDRESS: rec["address"],
                    schema.COUNTRY: rec["country"],
                }
            )
        rows.append(source_rows)

    truth = {}
    for k, ids in entity_to_ids[1].items():
        (s1_id,) = ids
        truth[s1_id] = sorted(entity_to_ids[2].get(k, []) + entity_to_ids[3].get(k, []))
    return rows, truth


def write_synthetic(out_dir: str | Path, cfg: SynthConfig | None = None) -> Path:
    """Write ``<out_dir>/train/*`` and ``<out_dir>/test/*`` in the challenge layout.

    Unlike the real data, the synthetic test split also gets a ground-truth
    file, so the test-time path can be scored end to end.
    """
    import csv

    cfg = cfg or SynthConfig()
    out_dir = Path(out_dir)
    rng = random.Random(cfg.seed)
    for split, n, countries in (
        ("train", cfg.train_entities, cfg.train_countries),
        ("test", cfg.test_entities, cfg.test_countries),
    ):
        rows, truth = generate_split(n, countries, cfg, rng)
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for source, source_rows in zip((1, 2, 3), rows):
            with open(
                split_dir / schema.source_file(split, source), "w", encoding="utf-8", newline="\n"
            ) as fh:
                writer = csv.writer(
                    fh, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n"
                )
                writer.writerow(schema.SOURCE_COLUMNS)
                for row in source_rows:
                    writer.writerow([row[c] for c in schema.SOURCE_COLUMNS])
        s1_order = [r[schema.ENTITY_ID] for r in rows[0]]
        write_id_lists(
            split_dir / schema.ground_truth_file(split),
            truth,
            s1_order,
            (schema.S1_ID, schema.MATCHED_IDS),
        )
    return out_dir
