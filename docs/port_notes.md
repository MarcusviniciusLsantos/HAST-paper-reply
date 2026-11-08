# Notas do port Java (MOA) → Python

Responsável: Marcus (fases 1–3). Este documento registra **o que foi portado, como foi validado e todas
as decisões/achados** que afetam a replicação do artigo (Nowak Assis et al., DMKD 2026).

## 1. Origem do código

| Componente | Origem | Situação no port |
| --- | --- | --- |
| HLAST, ARFHLAST, ARTEHLAST, LAST, EFLAST… | pasta `trees/` dos autores (Google Drive) | HT/HLAST/HLAST_D e variantes ARF/ARTE portados |
| AdaptiveRandomForest, ARTE | pasta `meta/` dos autores | portados |
| HoeffdingTree, observadores, InfoGain, NaiveBayes, HDDM_A, ADWIN, LeveragingBag, StreamingRandomPatches, ARFHoeffdingTree | MOA 2024.07 (fonte oficial) | portados |
| `ARTEAttributeClassObserver` | **não publicado** | **reconstruído** (ver §4.3) |
| ARE (AdaptiveRegularizedEnsemble) | **não publicado** | **reconstruído** (ver §4.4) |
| LADPU, Asfault | não disponíveis | pedidas aos autores; replicação com 11/13 bases reais |
| 24 fluxos sintéticos | não publicados (vêm do benchmark do KUE) | gerados com os geradores do MOA (ver §5) |

O código original fica intocado em `original_data_article/`. A cópia compilável usada como referência fica
em `java_ref/` (única alteração: o detector padrão `""` do `HLAST`/`LAST`/`EFLAST` virou `HDDM_A_Test`,
porque o MOA 2024.07 não aceita classe vazia como padrão; o artigo usa HDDM_A em todas as versões).

## 2. Estrutura do pacote `hast`

```
hast/core/jrandom.py        java.util.Random + MiscUtils.poisson (Python, bit a bit)
hast/core/fastrandom.py     o mesmo gerador em numba (estado = array int64[1])
hast/core/mstats.py         erf/erfc/normalProbability (Cephes, como no MOA), entropia
hast/core/arff.py           leitor ARFF denso (+ cache .npz)
hast/detectors/__init__.py  HDDM_A, ADWIN, ADWINChangeDetector (Python, legível)
hast/detectors/fast.py      ADWIN em numba (bit a bit igual ao Python)
hast/trees/hoeffding.py     HT e HLAST/HLAST_D em Python orientado a objetos (implementação de referência)
hast/trees/variants.py      variantes ARF/ARTE da implementação de referência
hast/trees/kernels.py       kernels numba das estatísticas de folha usados pela implementação de referência
hast/trees/fast.py          implementação rápida (arrays + numba) — usada nos experimentos
hast/ensembles/             LevBag, ARF, SRP, ARTE, ARE
hast/eval/prequential.py    avaliação prequencial (protocolo do artigo) e métricas
hast/eval/javaref.py        comparação instância a instância com a referência Java
hast/experiments.py         registro (ensemble × árvore) e CLI Java equivalente
hast/datasets.py            registro das bases com os nomes das tabelas do artigo
```

Há **duas implementações das árvores com resultados idênticos bit a bit**: a de objetos (`hoeffding.py`,
fácil de ler e modificar) e a rápida (`fast.py`, ~2–10× mais rápida no ensemble). A de objetos é o oráculo
dos testes da rápida. `make_model(..., impl="object"|"fast")` escolhe qual usar.

## 3. Validação

1. **Python × Java (MOA 2024.07 + código dos autores)**, instância a instância (votos exatos), via
   `java_ref/RefRunner` e `scripts/check_ensemble.py`:
   * árvores isoladas HT, HLAST, HLAST_D (Elec, 20 mil instâncias): 0 divergências de predição, mesmo
     número de nós e de splits;
   * LevBag, ARF, SRP, ARTE × HT, HLAST, HLAST_D (Elec, 5 mil instâncias, 10 membros): 0 divergências;
   * lote com 100 membros em CovType, Poker, INSECTS e Airlines: `results/equivalence/`.
2. **Árvore rápida × árvore de objetos**: bit a bit (votos, nº de nós, nº de splits por caminho) em Elec,
   Airlines e CovType, nas 9 combinações {single, ARF, ARTE} × {HT, HLAST, HLAST_D}.
3. **Protocolo**: ARF+HT no Outdoor reproduz o artigo (acc 63,490 ± 1,229 contra 63,487 ± 1,208).

**Única fonte de diferença numérica restante**: `Math.exp`/`Math.log` da JVM são intrínsecos da Intel e
diferem da libm em 1 ulp em ~0,3% / ~0,04% dos argumentos. Isso gera votos que diferem no último bit
(erro relativo < 1e-15), sem alterar predições nos testes; em fluxos longos pode, raramente, inverter um
empate e fazer as execuções divergirem. Por isso a replicação é avaliada por tendências e rankings.

## 4. Achados e decisões que afetam os resultados

### 4.1 Protocolo de avaliação (confirmado empiricamente)
* `EvaluatePrequential` com `BasicClassificationPerformanceEvaluator` (**cumulativo**, não janela);
* 20 pontos (`n // 20`); F1 do MOA = 2PR/(P+R) com P e R **macro**; uma classe ainda não predita/vista
  torna P (ou R) **NaN**, e o ponto é contado como **0** — por isso desvios como 85 ± 19 (CovType);
* média e **desvio populacional** dos 20 pontos.

### 4.2 Bugs/quirks do código original replicados de propósito
* **`ARFHLAST` observa `k−1` atributos por folha** (`for j < numAttributes - 1`), enquanto o
  `ARFHoeffdingTree` oficial observa `k`. O HLAST dentro do ARF fica em desvantagem no artigo.
  `ARFHLAST(fix_arf_subspace_bug=True)` corrige.
* **SRP (MOA)**: ao promover o learner de background num drift, o MOA não zera `bkgLearner`; a árvore
  promovida passa a ser treinada **duas vezes por instância** até o próximo warning (o ARF zera).
* **ARTE**: em splits nominais multiway cada ramo é criado com prob. 1/2, mas o contador de folhas soma
  todos os ramos (o tamanho reportado da árvore conta folhas inexistentes).
* **HLAST, caminho adaptativo**: não verifica pureza da folha (o periódico verifica).
* Gestão de memória do MOA (limite de 33 MB por árvore): reproduzida na lógica, mas o tamanho em bytes é
  uma **aproximação** do `SizeOf` da JVM. Só age quando o peso de treino acumulado bate exatamente um
  múltiplo de 10⁶ (HT) / 2·10⁶ (ensembles), o que quase nunca acontece com pesos Poisson.

### 4.3 `ARTEAttributeClassObserver` (reconstruído)
Não publicado. Pelo artigo do ARTE ("random cut points", estilo Extra-Trees): em cada tentativa de split,
um único ponto de corte uniforme em [min, max] do atributo na folha, sorteado com o `Random` da árvore;
Naive Bayes usa o estimador gaussiano. Implementado igualmente em Java (`java_ref/src/.../ARTEAttributeClassObserver.java`)
e em Python, para que a referência Java do ARTE use o mesmo observador.

### 4.4 ARE (reconstruído)
Não publicado. Implementado a partir da descrição do artigo HAST: ARF + seleção de instâncias (treina no
que erra; uma instância acertada só é usada após rejeitar `zeta` acertos da mesma classe) + detectores
alimentados só com instâncias de treino + votação apenas dos membros com acurácia ≥ média.
`zeta` não é informado: usamos 10. **Os números do ARE são aproximados.**

### 4.5 Hiperparâmetros das árvores dentro dos ensembles (suposição)
O artigo diz "parâmetros padrão do MOA". Usamos, para HLAST, os mesmos hiperparâmetros que cada ensemble
usa por padrão para a sua HT (LevBag: g=200, c=1e-7; ARF: g=50, c=0,01, e=2·10⁶; SRP: g=50, c=0,01;
ARTE: g=100, c=0,01, e=2·10⁶) e HDDM_A com padrões do MOA (drift 0,001, warning 0,005, bicaudal).

### 4.6 Discrepância encontrada com o artigo
O código publicado, no MOA 2024.07, **reproduz o artigo para ARF+HT** mas **não para ARF+HLAST/HLAST_D**
no Outdoor (ex.: HLAST_D acc 74,6 nosso/Java contra 67,6 no artigo). Python e Java concordam entre si,
logo a diferença vem da configuração/versão usada pelos autores, não do port. Variações de `g`, `c` e da
predição na folha não reproduziram os números do artigo. A comparação sistemática está no relatório de
replicação.

## 5. Fluxos sintéticos (suposições)
`scripts/generate_synthetic.py` gera os 24 fluxos com os geradores do MOA (10⁶ instâncias cada):
drifts igualmente espaçados (segmento = 10⁶/nº de conceitos) encadeados com `ConceptDriftStream`;
largura gradual = 10% do segmento; demais parâmetros padrão (AGR perturbação 5%; SEA ruído 10%; LED ruído
10% e 17 atributos irrelevantes, `LEDGeneratorDrift -d k`; RBF 50 centróides, 5 classes, 10 atributos,
velocidades 1e-5/1e-4/1e-3; HYPER 10 atributos, 2 com drift, magnitudes 1e-4/1e-3/1e-2).
`--seed-offset 1000` gera os **fluxos de calibração** (`data/calibration/`), com sementes diferentes, para a
calibração de γ, δ_ad e λ na fase 4 (nunca usar na avaliação).

## 6. Desempenho
* Compilação numba: ~1–2 min por processo (as funções que recebem a `jitclass` da árvore não entram no
  cache em disco). O executor usa workers de vida longa para compilar uma vez só.
* Vazão típica (100 membros): 3–6 ms por instância (CovType: ARTE-HLAST_D ≈ 27 min por execução completa;
  ARF-HLAST ≈ 41 min; SRP-HT ≈ 54 min; Airlines ARF ≈ 31 min).
* Memória: Airlines (nominais com 293 valores, splits multiway) chega a vários GB por execução; rode com
  pouco paralelismo.

### 4.7 F1 do MOA em classes raríssimas (Poker, Outdoor)
O F1 macro do MOA é indefinido enquanto alguma classe nunca foi predita. No Poker, a classe 9 tem 2 instâncias
em 829 mil; nas nossas execuções ela nunca é predita e o F1 vale 0 nos 20 pontos (artigo: ~48 ± 34), embora a
acurácia fique a menos de 1 ponto do artigo. Predizer essa classe uma única vez é questão de sorte e muda com
qualquer detalhe numérico. Na comparação com o artigo, olhar também a acurácia e a mediana das diferenças.
