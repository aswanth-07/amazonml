import pandas as pd
import pytest

from ber.normalize import NORMALIZED_COLUMNS, normalize_records
from ber.normalize.address import extract_postal_codes, normalize_address, normalize_country
from ber.normalize.name import normalize_name
from ber.normalize.text import clean, skeleton


def test_clean_basics():
    assert clean("Société Générale") == "societe generale"
    assert clean("McDonald's") == "mcdonalds"
    assert clean("A&B L.L.C.") == "a and b llc"
    assert clean("  Hello,\tWorld!! ") == "hello world"
    assert clean("") == ""


def test_transliteration_of_non_latin_script():
    assert clean("श्री") != ""  # Devanagari becomes Latin letters rather than vanishing
    assert clean("श्री").isascii()


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("ABC Corp", "ABC Corporation"),
        ("Acme Pvt. Ltd.", "Acme Private Limited"),
        ("Smith & Sons LLC", "Smith and Sons L.L.C."),
        ("Dupont et Fils SARL", "Dupont & Fils S.A.R.L."),
        ("The Golden Gate Bakery Inc", "Golden Gate Bakery Incorporated"),
    ],
)
def test_name_variants_share_core(a, b):
    na, nb = normalize_name(a), normalize_name(b)
    assert na.core == nb.core
    assert na.legal == nb.legal


def test_transliteration_skeleton():
    a = normalize_name("Sri Lakshmi Venkateswara Traders")
    b = normalize_name("Shree Laxmi Venkateshwara Traders")
    assert a.skeleton == b.skeleton
    assert skeleton("ganesha") == skeleton("ganesh")


def test_dba_alias_and_acronym():
    n = normalize_name("Northwind Holdings LLC DBA Blue Cafe")
    assert n.core == "northwind holdings"
    assert n.alias == "blue cafe"
    assert normalize_name("International Business Machines Corp").acronym == "ibm"


def test_name_of_only_a_legal_form_is_kept():
    assert normalize_name("The Company").core != ""


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("12 Main St., Austin", "12 Main Street, Austin"),
        ("12 avenue Victor Hugo, Lyon", "12 av. Victor Hugo, Lyon"),
        ("5 MG Road, Bangalore", "5 M.G. Rd, Bengaluru"),
        ("No. 5, MG Road, Bengaluru", "5, MG Road, Bengaluru"),
    ],
)
def test_address_variants_share_core(a, b):
    na, nb = normalize_address(a), normalize_address(b)
    assert na.core.replace("m g", "mg") == nb.core.replace("m g", "mg")


def test_postal_codes():
    assert extract_postal_codes("MG Road, Bengaluru - 560 001") == ["560001"]
    assert extract_postal_codes("1 Main St, Seattle, WA 98101-1234") == ["98101"]
    assert extract_postal_codes("3 rue de Paris, 75001 Paris") == ["75001"]
    # a leading 5-digit number is a house number, not a ZIP
    assert extract_postal_codes("10500 NE 8th St, Bellevue, WA 98004") == ["98004"]
    assert extract_postal_codes("10500 NE 8th St, Bellevue") == []


def test_landmarks_are_separated_from_core():
    a = normalize_address("No. 12, MG Road, Near SBI ATM, Bengaluru 560001")
    assert a.landmark == "near sbi atm"
    assert "sbi" not in a.core
    assert a.postal == "560001"
    assert "560001" not in a.core
    assert a.numbers == "12"


def test_country_is_an_open_set():
    assert normalize_country("US") == normalize_country("United States") == normalize_country("USA")
    assert normalize_country("India") == "india"
    assert normalize_country("France") == "france"
    assert normalize_country("Germany") == "germany"  # never seen, passes through
    assert normalize_country("") == ""


def test_normalize_records_adds_all_columns():
    df = pd.DataFrame(
        {
            "entity_id": ["S1-1"],
            "business_name": ["Café de la Gare SARL"],
            "business_address": ["3 place de la Gare, 69002 Lyon"],
            "country": ["France"],
        }
    )
    out = normalize_records(df)
    for col in NORMALIZED_COLUMNS:
        assert col in out
    assert out.loc[0, "name_core"] == "cafe de gare"
    assert out.loc[0, "postal"] == "69002"
