"""Find frozen request/complaint cues in the selected global-to-local NMF sentences."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml

from topic_requirement_review import LEXICON, score_lexical


ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "03-topic-modeling/configs/params.yaml"


def search_terms(global_run: Path, output_dir: Path, topics: list[int]):
    frames = []
    for topic in topics:
        source = global_run / "restricted_gensim" / f"topic{topic:02d}" / "nmf_results.csv"
        if not source.is_file():
            raise FileNotFoundError(source)
        frame = pd.read_csv(source, keep_default_na=False)
        if not frame.global_topic_id.eq(topic).all():
            raise ValueError(f"ID global divergente em {source}")
        frames.append(frame)
    evidence = pd.concat(frames, ignore_index=True)
    if not evidence.sent_id.is_unique:
        raise ValueError("sent_id duplicado entre temas NMF")

    lexical = score_lexical(evidence.sentence)
    evidence = pd.concat([evidence, lexical], axis=1)
    match_rows = []
    counts = []
    for kind, patterns in LEXICON.items():
        for pattern in patterns:
            matched = evidence.sentence.str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False)
            counts.append({"tipo": kind, "termo_regex": pattern, "n_frases": int(matched.sum())})
            match_rows.append((kind, pattern, matched))
    evidence["termos_encontrados"] = [
        json.dumps([{"tipo": kind, "termo_regex": pattern}
                    for kind, pattern, matches in match_rows if matches.iloc[i]],
                   ensure_ascii=False)
        for i in range(len(evidence))
    ]
    if not evidence.lex_n.eq(evidence.termos_encontrados.map(lambda s: len(json.loads(s)))).all():
        raise ValueError("contagem lexical divergente do lexico congelado")

    def aggregate(columns):
        result = evidence.groupby(columns, dropna=False).agg(
            n_frases=("sent_id", "size"),
            n_com_termo=("lex_hit", "sum"),
            n_pedido=("intent_hint", lambda x: int(x.eq("pedido").sum())),
            n_queixa=("intent_hint", lambda x: int(x.eq("queixa").sum())),
        ).reset_index()
        result["pct_com_termo"] = (100 * result.n_com_termo / result.n_frases).round(2)
        return result

    by_global = aggregate(["global_topic_id"])
    by_local = aggregate(["global_topic_id", "local_topic_id"])
    if int(by_global.n_frases.sum()) != len(evidence) or int(by_global.n_com_termo.sum()) != int(evidence.lex_hit.sum()):
        raise ValueError("resumo global nao fecha com as evidencias")
    if int(by_local.n_frases.sum()) != len(evidence) or int(by_local.n_com_termo.sum()) != int(evidence.lex_hit.sum()):
        raise ValueError("resumo local nao fecha com as evidencias")

    summary = {
        "global_run": str(global_run.resolve()),
        "topics": list(topics),
        "n_sentences": len(evidence),
        "n_lex_hit": int(evidence.lex_hit.sum()),
        "pct_lex_hit": round(100 * float(evidence.lex_hit.mean()), 2),
        "n_pedido": int(evidence.intent_hint.eq("pedido").sum()),
        "n_queixa": int(evidence.intent_hint.eq("queixa").sum()),
        "n_without_local_topic": int(evidence.local_topic_id.eq(-1).sum()),
        "lexicon": LEXICON,
        "unit": "sentence; a lexical hit is not a human-confirmed requirement",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(output_dir / "evidencias.csv", index=False, encoding="utf-8")
    evidence.loc[evidence.lex_hit].to_csv(output_dir / "frases_com_termo.csv", index=False, encoding="utf-8")
    by_global.to_csv(output_dir / "sinais_por_topico_global.csv", index=False, encoding="utf-8")
    by_local.to_csv(output_dir / "sinais_por_subtopico.csv", index=False, encoding="utf-8")
    pd.DataFrame(counts).to_csv(output_dir / "contagem_por_termo.csv", index=False, encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    with PARAMS.open(encoding="utf-8") as handle:
        final = yaml.safe_load(handle)["final_sentence_pipeline"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("global_run", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--topics", type=int, nargs="+", default=final["selected_topics"])
    args = parser.parse_args()
    if len(set(args.topics)) != len(args.topics):
        parser.error("--topics contem IDs duplicados")
    print(json.dumps(search_terms(args.global_run, args.output_dir, args.topics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
