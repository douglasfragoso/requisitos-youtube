"""Selecao STM para os corpora YouTube.

    .venv/Scripts/python.exe 03-topic-modeling/notebooks/_selecao.py stm youtube_doc

Nao ajusta modelo: le o CSV de grade que ja existe e decide em segundos.
A selecao usa fronteira de Pareto e desempate declarado por menor K.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd


def _forcar_utf8_no_stdout() -> None:
    """Reconfigura o stdout para UTF-8 — chamado somente pelo ``__main__``."""
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "data" / "output"
SEED = 42

FAIXA_K = {
    # youtube_doc: ~1.100 transcricoes em 9 categorias de produto; abaixo de 8
    # topicos nao ha um aspecto por familia; acima de 25 fragmenta.
    "youtube_doc": (8, 25),
    # youtube_sent (sentenca + sentimento) fica fora do escopo por ora; volta
    # quando existir o pipeline de sentenca.

}

DESEMPATE_STM = "K"
DESEMPATE_STM_ASCENDING = True


def _pareto(df: pd.DataFrame, eixos: list[str], verbose: bool = False) -> pd.DataFrame:
    """Pontos nao dominados: nenhum outro e >= em todos os eixos e > em algum."""
    eixos = [e for e in eixos if e in df.columns and df[e].notna().any()]
    if not eixos or df.empty:
        return df.iloc[0:0]
    completo = df.dropna(subset=eixos)
    n_fora = len(df) - len(completo)
    if n_fora and verbose:
        print(f"   ({n_fora} configuracoes fora da fronteira por NaN em "
              f"{', '.join(eixos)} — nao medidas, nao comparaveis)")
    if completo.empty:
        return df.iloc[0:0]
    valores = completo[eixos].to_numpy()
    manter = [completo.index[i] for i in range(len(valores))
              if not (((valores >= valores[i]).all(axis=1)) &
                      ((valores > valores[i]).any(axis=1))).any()]
    return completo.loc[manter]


def _eixos_stm(d: pd.DataFrame) -> list[str]:
    """Usa Diversity como terceiro eixo quando a grade a disponibiliza."""
    base = ["semantic_coherence", "exclusivity_stm"]
    if "diversity" in d.columns and d["diversity"].notna().any():
        return base + ["diversity"]
    return base


def selecionar_stm(corpus: str) -> None:
    """Seleciona K do STM pela fronteira e por menor K em caso de empate."""
    base = OUT / corpus / "stm"
    diag_p = base / "stm_grid_k_diagnostics.csv"
    if not diag_p.exists():
        raise SystemExit(f"ausente: {diag_p}")

    d = pd.read_csv(diag_p).rename(columns={"k": "K"})
    k_min, k_max = FAIXA_K[corpus]
    eixos_stm = _eixos_stm(d)

    print("=" * 78)
    print(f"SELECAO — STM / {corpus}   (faixa de K declarada: [{k_min},{k_max}])")
    print("=" * 78)
    if "diversity" not in eixos_stm:
        print("   !! sem coluna 'diversity' — fronteira de 2 eixos")

    faltando = [c for c in eixos_stm if c not in d.columns or not d[c].notna().any()]
    if faltando:
        print(f"\n!! stm_grid_k_diagnostics.csv sem {faltando}.")
        return

    adm = d[(d.K >= k_min) & (d.K <= k_max)].copy()
    print("\nCascata de admissibilidade:")
    print(f"   grade completa: {len(d)}")
    print(f"   apos A3  K em [{k_min},{k_max}]: {len(adm)}")
    if adm.empty:
        print("\nNENHUM K admissivel na grade.")
        return

    fr = _pareto(adm, eixos_stm, verbose=True)
    if fr.empty:
        print("\nFronteira vazia — nenhum K admissivel tem os eixos medidos.")
        return

    print(f"\nFronteira de Pareto ({' x '.join(eixos_stm)}): {len(fr)} de {len(adm)}")
    criterios = ["cv"] + eixos_stm + [DESEMPATE_STM]
    escolhido = fr.sort_values(criterios, ascending=[False] * (len(criterios) - 1) + [True]).iloc[0]
    print("Decisao: maior C_v; empates por coerencia, exclusividade, diversity e menor K")

    cols = ["K", "cv", "semantic_coherence", "exclusivity_stm", "diversity",
            "heldout_likelihood", "residual_dispersion"]
    cols = [c for c in cols if c in fr.columns]
    pd.set_option("display.width", 220)
    print("\n--- fronteira (ordenada por K) ---")
    print(fr.sort_values(DESEMPATE_STM, ascending=DESEMPATE_STM_ASCENDING)[cols]
          .round(4).to_string(index=False))
    print(f"\n>>> ESCOLHIDO: K={int(escolhido['K'])}")

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("modelo", choices=["stm"])
    ap.add_argument("corpus", choices=sorted(FAIXA_K))
    args = ap.parse_args()
    selecionar_stm(args.corpus)
    return 0


if __name__ == "__main__":
    _forcar_utf8_no_stdout()
    raise SystemExit(main())