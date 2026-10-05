"""Evaluate STM and global NMF topic words on the same sentence corpus.

This is a post-hoc comparison. It does not reproduce either model's training
preprocessing or turn topic coherence into an elicitation metric.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from nltk.stem.snowball import SnowballStemmer


ROOT = Path(__file__).resolve().parents[2]
STM = ROOT / "03-topic-modeling/data/output/youtube_sent/stm/youtube_sent_20260921_190124"
NMF = ROOT / "03-topic-modeling/data/output/youtube_sent/nmf_global/nmf_global_sentence_k20_20261003_134035"
TOKEN_RE = re.compile(r"[a-z]+")


def main(output_dir: Path):
    stem = SnowballStemmer("english").stem
    stm = pd.read_csv(STM / "stm_results.csv", usecols=["post_id", "text"],
                      dtype={"post_id": str}, keep_default_na=False)
    nmf_ids = set(pd.read_csv(NMF / "nmf_results.csv", usecols=["sent_id"],
                              dtype={"sent_id": str}).sent_id)
    stm = stm.loc[stm.post_id.isin(nmf_ids)]
    if not stm.post_id.is_unique or len(stm) != 114888:
        raise ValueError("o corpus compartilhado nao corresponde ao run STM pinado")

    texts = [[stem(token) for token in TOKEN_RE.findall(sentence.lower())]
             for sentence in stm.text]
    dictionary = Dictionary(texts)
    vocabulary = set(dictionary.token2id)
    source_frames = {
        "STM_global": pd.read_csv(STM / "stm_topics_for_eval.csv"),
        "NMF_global": pd.read_csv(NMF / "topics.csv"),
    }
    rows, topic_words = [], []
    for model, source in source_frames.items():
        field = "keywords" if model == "STM_global" else "top_terms"
        for record in source.itertuples(index=False):
            raw_words = getattr(record, field).split(", ")
            normalized = []
            for word in raw_words:
                token_match = TOKEN_RE.findall(word.lower())
                if not token_match:
                    continue
                token = stem(token_match[0])
                if token in vocabulary and token not in normalized:
                    normalized.append(token)
                if len(normalized) == 10:
                    break
            if len(normalized) < 2:
                raise ValueError(f"menos de dois termos avaliaveis: {model} {record.topic_id}")
            rows.append({"model": model, "topic_id": int(record.topic_id),
                         "n_terms": len(normalized),
                         "evaluated_terms": ", ".join(normalized)})
            topic_words.append(normalized)

    coherence = CoherenceModel(topics=topic_words, texts=texts,
                               dictionary=dictionary, coherence="c_npmi",
                               topn=10, window_size=10, processes=1)
    per_topic = coherence.get_coherence_per_topic()
    result = pd.DataFrame(rows)
    result["npmi_shared_corpus"] = per_topic
    summary = result.groupby("model").agg(
        n_topics=("topic_id", "size"),
        npmi_mean=("npmi_shared_corpus", "mean"),
        npmi_median=("npmi_shared_corpus", "median"),
        min_terms=("n_terms", "min"),
        max_terms=("n_terms", "max"),
    ).reset_index()
    summary["topic_diversity_evaluated_terms"] = summary.model.map({
        model: len(set(word for terms in result.loc[result.model.eq(model), "evaluated_terms"]
                       for word in terms.split(", "))) / int(result.loc[result.model.eq(model), "n_terms"].sum())
        for model in source_frames
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "npmi_global_por_topico.csv", index=False, encoding="utf-8")
    summary.to_csv(output_dir / "npmi_global_resumo.csv", index=False, encoding="utf-8")
    metadata = {
        "n_shared_sentences": len(texts),
        "sentence_source": str(STM / "stm_results.csv"),
        "topic_sources": {key: str(STM if key == "STM_global" else NMF) for key in source_frames},
        "tokenization": "lowercase [a-z]+, English Snowball stemming, no stopword removal",
        "coherence": "Gensim c_npmi, sliding window 10, top 10 distinct normalized words per topic",
        "comparison_scope": "post-hoc lexical coherence on the same original sentences; not model training coherence or requirement precision",
    }
    (output_dir / "npmi_global_metodo.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    main(args.output_dir)
