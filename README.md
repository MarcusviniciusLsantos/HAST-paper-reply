# HAST em Python — replicação e regularização do split adaptativo

Port para Python das **Hoeffding Adaptive Splitting Trees** (HLAST/HLAST_D) e dos ensembles usados em
Nowak Assis, Barddal & Enembreck, *Hoeffding adaptive splitting trees for data stream classification with
concept drift and ensemble learning*, DMKD 40:91 (2026) — projeto final da disciplina *Aprendizado de Máquina
em ambiente não estacionário* (Íris, Marcus, Sérgio).

* **O que está pronto (fases 1–2, Marcus):** HT, HLAST, HLAST_D, HDDM_A, ADWIN e os 5 ensembles
  (LevBag, ARF, SRP, ARTE e ARE reconstruído), avaliação prequencial com o protocolo do artigo, as 11
  bases reais disponíveis, os 24 fluxos sintéticos e 24 fluxos de calibração, e testes contra o Java.
* **Fidelidade:** LevBag, ARF, SRP e ARTE × {HT, HLAST, HLAST_D} reproduzem o código Java dos autores
  (MOA 2024.07) **instância a instância** (0 divergências de predição nos testes). Detalhes, achados e
  suposições em [docs/port_notes.md](docs/port_notes.md).

## Instalação

```bash
uv venv .venv && uv pip install -e ".[dev]"
```

Os dados (`data/`) não são versionados. As bases reais estão no site do artigo
(<https://sites.google.com/view/last-ensemble>): coloque os `.arff` em `data/real/` com os nomes de
`hast/datasets.py`. Os fluxos sintéticos, de calibração e desbalanceados são gerados com os geradores do MOA
(requer o `moa.jar` e um JDK, ver *Referência Java*):

```bash
python scripts/generate_synthetic.py -j 4                                   # data/synthetic (avaliação)
python scripts/generate_synthetic.py --seed-offset 1000 --out data/calibration -j 4   # calibração (fase 4)
python scripts/prepare_imbalanced.py                                        # data/imbalanced
```

## Uso

```python
from hast.core.arff import load_arff
from hast.trees.hoeffding import Schema
from hast.experiments import make_model
from hast.eval.prequential import run_prequential

s = load_arff("data/real/elecNormNew.arff")
m = make_model("ARTE", "HLAST_D", ensemble_size=100)   # ensembles: LevBag ARF SRP ARTE ARE
m.set_schema(Schema.from_header(s.header))              # árvores:  HT HLAST HLAST_D
r = run_prequential(m, s)
print(r.summary())   # F1/acc/kappa/G-mean (média ± desvio dos 20 pontos), tamanho das árvores, splits por caminho, CPU
```

Experimentos em lote (retomáveis; um JSON por execução em `results/raw/`):

```bash
python scripts/run_experiments.py --datasets real --ensembles ARTE SRP --trees HT HLAST HLAST_D -j 4
python scripts/run_experiments.py --datasets synthetic --shard 0/3 -j 8    # máquina 1 de 3
```

> A primeira chamada de cada processo compila o numba (~1–2 min). Airlines usa vários GB por execução
> (nominais com 293 valores): rode-o com `-j 1` e `--fresh-workers`.

### Replicação completa numa máquina local

```bash
nohup setsid ./scripts/run_replication_queue.sh > results/raw/queue.log 2>&1 &
```

4 filas com jobs disjuntos, 1 worker cada (no i5-12450H, 4 execuções simultâneas rendem ~3,6×; com 6 a vazão
total cai pela metade). Estimativa medida: ~175 h de CPU no total, **~2 dias de relógio**. Deixe o notebook na
tomada, sem suspensão, e uma janela do WSL aberta. O progresso fica em `results/raw/queue.log` e
`results/raw/run.log`; a fila é retomável (execuções concluídas são puladas).

## Testes

```bash
python -m pytest tests/test_core.py          # rápido: RNG/estatística/detectores/métricas/ARFF
python -m pytest tests/test_equivalence.py   # lento: árvore rápida == árvore de objetos; Python == Java
```

## Referência Java (só local, não versionada)

O código Java dos autores (`original_data_article/`) e a cópia compilável usada para validar o port
(`java_ref/`, com o `RefRunner` que grava os votos de cada instância) **não vão para o repositório**. Os
testes contra o Java são pulados quando `java_ref/` não existe. Para refazer a validação, é preciso o código
dos autores (site do artigo), o MOA 2024.07 e um JDK em `third_party/`:

```bash
# MOA: https://github.com/Waikato/moa/releases/download/2024.07.0/moa-release-2024.07.0-bin.zip
#      -> copiar lib/moa.jar e lib/sizeofag-1.1.0.jar para java_ref/lib/
# JDK 21: https://adoptium.net  -> link simbólico third_party/jdk
java_ref/run_ref.sh "meta.AdaptiveRandomForest -s 10" data/real/elecNormNew.arff 5000 java_ref/out/arf.txt
python scripts/check_ensemble.py ARF HT data/real/elecNormNew.arff 5000 10
```

## Para a fase 4 (regularização, Íris)

O critério adaptativo `G(Xa) > 0` (linha 18 do Algoritmo 1) é um método Python isolado. Basta sobrescrevê-lo:

```python
from hast.trees.fast import FastTree
from hast.experiments import make_model

class HLASTR1(FastTree):
    """R1: ganho mínimo γ no caminho adaptativo."""
    def __init__(self, gamma=0.01, **kw):
        super().__init__(**kw); self.gamma = gamma
    def params(self):                      # necessário para clone() nos ensembles
        p = super().params(); p["gamma"] = self.gamma; return p
    def adaptive_split_accepts(self, node, suggestions):
        # suggestions: [(merit, kind, att, value, dists, lens), ...] em ordem crescente de mérito
        return suggestions[-1][0] > self.gamma      # self.depth_of(node) disponível para R3

m = make_model("ARTE", "HLAST_D", tree_cls=HLASTR1, tree_kw={"gamma": 0.01})
```

Contadores de diagnóstico por árvore: `n_attempts_periodic/adaptive`, `n_splits_periodic/adaptive`
(agregados em `RunResult.splits`). Os fluxos de calibração estão em `data/calibration/`.

## Revisão de segurança do material dos autores

Antes de executar qualquer coisa, todo o material copiado do site dos autores foi inspecionado:
16 arquivos `.java` (só lógica do MOA; nenhum uso de rede, processos externos, reflexão, escrita de
arquivos ou `Runtime.exec` — o `System.exit` do `LAST.java` é um tratamento de erro), 11 ARFF (texto puro,
sem linhas anômalas nem caracteres não-ASCII), 4 PDFs (sem JavaScript, ações automáticas ou anexos) e um
script Python (`generate_synthetic_streams.py`: só monta comandos do MOA; importa `smtplib` sem usá-lo e
depende de um módulo ausente — não foi executado). Os arquivos `*:Zone.Identifier` são metadados do
Windows (marca de download) e podem ser apagados.
