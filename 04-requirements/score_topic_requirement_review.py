"""Score human annotations for the final NMF run or the historical STM baseline."""

import argparse
import json
from pathlib import Path

import pandas as pd

from topic_requirement_review import score_annotated_sample


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", required=True, type=Path)
parser.add_argument("--annotations", required=True, type=Path,
                    help="Copy of amostra_cega.csv after human review")
args = parser.parse_args()
metadata = json.loads((args.run / "metadata.json").read_text(encoding="utf-8"))
annotations = pd.read_csv(args.annotations)
origin = pd.read_csv(args.run / "amostra_origem.csv")
results, by_topic = score_annotated_sample(
    annotations, origin, top_k=int(metadata["top_k"]),
    n_random=int(metadata["n_random"]),
)
payload = {"status": "anotado", "n_reviewed": len(annotations),
           "precision_by_ranking": results,
           "note": "Contagens por topico descrevem o pool amostrado, nao a prevalencia no corpus."}
(args.run / "resultado_revisao.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
by_topic.to_csv(args.run / "confirmados_por_topico_na_amostra.csv",
                index=False, encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
