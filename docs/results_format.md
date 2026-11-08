# Formato dos resultados

Cada execução (base × ensemble × árvore) gera **um JSON** em `results/raw/`, com nome
`<base>__<ensemble>__<árvore>[<tag>].json` (parênteses e vírgulas do nome da base viram `_` e `-`).
O arquivo é escrito por `RunResult.save` (`hast/eval/prequential.py`) só no fim da execução; arquivos
existentes são pulados, o que torna o executor retomável.

| Campo | Conteúdo |
| --- | --- |
| `dataset`, `ensemble`, `tree` | identificação (`tree` ∈ HT, HLAST, HLAST_D; variantes da fase 4 usam `--tag`) |
| `n_instances` | instâncias processadas |
| `sample_points` | os 20 pontos de amostragem (contagem de instâncias) |
| `metrics.accuracy/kappa/f1/precision/recall/gmean` | lista com o valor (%) em cada ponto, métricas cumulativas do MOA; `NaN` quando indefinido |
| `confusion` | matriz de confusão cumulativa `[verdadeira][predita]` em cada ponto — permite derivar qualquer métrica depois |
| `extra.recall_per_class` | recall (%) por classe em cada ponto (classe minoritária nas bases desbalanceadas) |
| `tree_size_mean`, `tree_size_std` | média e desvio do nº de nós das árvores do ensemble em cada ponto |
| `splits.splits_periodic`, `splits.splits_adaptive` | splits realizados por caminho, somados nas árvores ativas no fim |
| `cpu_seconds`, `wall_seconds`, `peak_rss_mb` | custo (CPU do processo, tempo de parede, pico de memória residente do worker) |
| `extra` | tamanho do ensemble, implementação, kwargs, nº de drifts, contadores do ARE, etc. |

**Resumo como no artigo:** `hast.eval.prequential.summarize(lista)` → (média, desvio populacional) com
`NaN` contado como 0. `RunResult.summary()` aplica isso a acurácia, F1, kappa e G-mean.

**Comparação com o artigo:** `scripts/parse_paper_tables.py` → `results/paper/paper_results.csv`
(métrica, ensemble, base, árvore, média, desvio — 37 bases × 5 ensembles × 8 árvores, F1 e acurácia);
`scripts/replication_report.py` → `results/replication/comparison.csv` e `summary.json`.

Observações para a análise (fase 6):
* G-mean aqui = média geométrica dos recalls das classes já vistas (não é métrica do MOA).
* `peak_rss_mb` é do processo worker, que pode ter rodado execuções anteriores (é um limite superior);
  para comparar custo, prefira `cpu_seconds`.
* Os nomes das bases seguem as tabelas do artigo (`hast/datasets.py`); os LED recorrentes aparecem
  no artigo sem o "r" (`LED-f(7,5,3,1,3,5,7)a`), o que `datasets.paper_name` já trata.
