import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_corpus import COLUNAS, build_corpus, detect_product_category, load_overrides  # noqa: E402


def _rec(vid="abcdefghijk", titulo="Dell XPS 13 Review", transcricao="word " * 50,
         fonte="whisper", upload="20240620", channel="Dave2D", meta=True):
    r = {"url": f"https://youtu.be/{vid}", "titulo": titulo, "duracao": "10:00",
         "transcricao": transcricao, "fonte": fonte}
    if meta:
        r["youtube_metadata"] = {"id": vid, "channel": channel, "upload_date": upload,
                                 "view_count": 1000, "like_count": 50, "comment_count": 7,
                                 "language": "en"}
    return r


@pytest.mark.parametrize("titulo,esperado", [
    ("Dell XPS 13 Review", "laptop"),
    ("MacBook Air M3 Review: Finally!", "laptop"),
    ("Alienware 15 R3 Review (GTX 1070)", "laptop"),
    ("Samsung Galaxy Book4 Pro 14: Why Does This Exist?", "laptop"),
    ("iPhone 15 Pro Max Review", "phone"),
    ("Nokia Lumia 1020 Review!", "phone"),
    ("Pixel 8 Review", "phone"),
    ("iPad Pro M4 Review", "tablet"),
    ("Sony WH-1000XM5 Review", "headphone"),
    ("AirPods Pro 2 Review", "headphone"),
    ("Apple Watch Ultra Review", "watch"),
    ("Samsung S95D OLED TV Review", "tv"),
    ("Steam Deck OLED Review", "console"),
    ("Mac Studio M2 Ultra Review", "desktop"),
    ("Apple Vision Pro Review", "vr"),
    ("Ray-Ban Meta Gen 2 Review", "vr"),
    ("RED Hydrogen One Review", "phone"),
    ("Motorola Droid Turbo Review", "phone"),
    ("Redmi Note 15 Pro Review", "phone"),
    ("Sony RX100 IV Review", "camera"),
    ("Canon 24mm f/1.4 L Review", "camera"),
    ("LG UltraFine 5K Review", "monitor"),
    ("Audio Technica ATH-M70X Review", "headphone"),
    ("Samsung Galaxy Gear Review", "watch"),
    ("Asus Transformer Pad Infinity Review", "tablet"),
    ("Lenovo LOQ Review", "laptop"),
    ("Alienware Aurora 2025 Review", "desktop"),
    ("LG Optimus G Pro Review", "phone"),
    ("Samsung Galaxy Round Review", "phone"),
    ("Asus M16 Review", "laptop"),
    ("Asus ProArt PX 13 Review", "laptop"),
    ("Google Stadia Review", "console"),
    ("Tesla Model 3 Review", "other"),
    ("Studio Display Review", "monitor"),
    ("My Personal Setup from 2019", "other"),
])
def test_detect_product_category(titulo, esperado):
    assert detect_product_category(titulo) == esperado


def test_override_vence_o_regex():
    df = build_corpus([_rec(titulo="Glass is glass")], overrides={"abcdefghijk": "phone"})
    assert df.loc[0, "product_category"] == "phone"


def test_colunas_e_tipos():
    df = build_corpus([_rec()], overrides={})
    assert list(df.columns) == COLUNAS
    assert df.loc[0, "post_id"] == "abcdefghijk"
    assert df.loc[0, "upload_date"] == "2024-06-20"
    assert df.loc[0, "year"] == 2024
    assert df.loc[0, "source"] == "whisper"
    assert df.loc[0, "language_meta"] == "en"
    assert bool(df.loc[0, "title_has_review"]) is True


def test_descarta_sem_transcricao_ou_sem_metadados():
    recs = [_rec(vid="a" * 11), _rec(vid="b" * 11, transcricao=""), _rec(vid="c" * 11, meta=False)]
    df = build_corpus(recs, overrides={})
    assert df["post_id"].tolist() == ["a" * 11]


def test_post_id_unico_levanta_em_duplicata():
    with pytest.raises(ValueError, match="post_id duplicado"):
        build_corpus([_rec(), _rec()], overrides={})


def test_load_overrides_le_csv(tmp_path):
    p = tmp_path / "ov.csv"
    p.write_text("post_id,product_category\nabcdefghijk,phone\n", encoding="utf-8")
    assert load_overrides(p) == {"abcdefghijk": "phone"}
    assert load_overrides(tmp_path / "inexistente.csv") == {}