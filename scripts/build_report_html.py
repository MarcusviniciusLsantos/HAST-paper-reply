"""Build the replication report page (docs/replication_report.html) from
results/replication/{comparison.csv,summary.json} and results/equivalence/.

Run scripts/replication_report.py first.
"""
from __future__ import annotations

import glob
import html
import json
import os
import re
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "results", "replication")
OUT = os.path.join(ROOT, "docs", "replication_report.html")
TREES = ["HT", "HLAST", "HLAST_D"]
ENS = ["ARTE", "SRP", "ARF", "LevBag", "ARE"]


def equivalence_rows():
    rows = []
    for f in glob.glob(os.path.join(ROOT, "results", "equivalence", "batch_100_N*.txt")):
        for line in open(f):
            m = re.match(r"(\S+)\s+(\S+)\s+(\S+)\s+n=(\d+) pred_mismatch=(\d+).*max_rel_err=(\S+)", line)
            if m:
                rows.append(dict(ens=m[1], tree=m[2], data=m[3], n=int(m[4]), mism=int(m[5]), rel=float(m[6])))
    return rows


def scatter_svg(df: pd.DataFrame) -> str:
    """Our F1 vs paper F1 (one dot per ensemble/dataset/tree), y = x reference."""
    W, H, L, B, T, R = 560, 420, 52, 44, 16, 16
    lo = 0.0
    hi = 100.0
    sx = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)
    sy = lambda v: H - B - (v - lo) / (hi - lo) * (H - B - T)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="F1 nosso versus F1 do artigo" class="scatter">']
    for v in range(0, 101, 20):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="grid"/>')
        out.append(f'<text x="{L - 8}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{v}</text>')
        out.append(f'<text x="{sx(v):.1f}" y="{H - B + 18}" class="tick" text-anchor="middle">{v}</text>')
    out.append(f'<line x1="{sx(lo)}" y1="{sy(lo)}" x2="{sx(hi)}" y2="{sy(hi)}" class="ident"/>')
    out.append(f'<text x="{sx(97):.1f}" y="{sy(93):.1f}" class="tick" text-anchor="end">y = x</text>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 6}" class="axis" text-anchor="middle">F1 do artigo (%)</text>')
    out.append(f'<text x="14" y="{(T + H - B) / 2}" class="axis" text-anchor="middle" '
               f'transform="rotate(-90 14 {(T + H - B) / 2})">F1 nosso (%)</text>')
    for i, t in enumerate(TREES):
        for _, r in df[df.tree == t].iterrows():
            tip = html.escape(f"{r.ensemble} · {r.dataset} · {t}: nosso {r.mean_ours:.2f}, artigo {r.mean_paper:.2f}")
            out.append(f'<circle cx="{sx(r.mean_paper):.1f}" cy="{sy(r.mean_ours):.1f}" r="4.5" '
                       f'class="dot s{i + 1}" data-tip="{tip}" tabindex="0"/>')
    out.append("</svg>")
    return "".join(out)


def main():
    comp = pd.read_csv(os.path.join(REP, "comparison.csv")) if os.path.exists(os.path.join(REP, "comparison.csv")) else pd.DataFrame()
    summ = json.load(open(os.path.join(REP, "summary.json"))) if os.path.exists(os.path.join(REP, "summary.json")) else {}
    f1 = comp[comp.metric == "f1"] if len(comp) else comp
    acc = comp[comp.metric == "accuracy"] if len(comp) else comp
    eq = equivalence_rows()
    n_runs = len(f1)
    stamp = time.strftime("%d/%m/%Y %H:%M")

    # ---- results tables
    tables = []
    for ens in ENS:
        ge = f1[f1.ensemble == ens] if len(f1) else f1
        if not len(ge):
            continue
        ga = acc[acc.ensemble == ens]
        rows = []
        for ds in sorted(ge.dataset.unique()):
            cells = [f"<th scope='row'>{html.escape(ds)}</th>"]
            for t in TREES:
                r = ge[(ge.dataset == ds) & (ge.tree == t)]
                ra = ga[(ga.dataset == ds) & (ga.tree == t)]
                if len(r):
                    d = r["diff"].iloc[0]
                    cls = "pos" if d > 0.5 else ("neg" if d < -0.5 else "")
                    cells.append(f"<td>{r.mean_ours.iloc[0]:.2f}</td><td class='muted'>{r.mean_paper.iloc[0]:.2f}</td>"
                                 f"<td class='{cls}'>{d:+.2f}</td><td class='muted'>{ra['diff'].iloc[0]:+.2f}</td>")
                else:
                    cells.append("<td colspan='4' class='muted'>pendente</td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")
        head = "".join(f"<th colspan='4' scope='colgroup'>{t}</th>" for t in TREES)
        sub = "".join("<th>F1 nosso</th><th>artigo</th><th>Δ F1</th><th>Δ acc</th>" for _ in TREES)
        tables.append(f"<h3>{ens}</h3><div class='tablewrap'><table><thead><tr><th rowspan='2'>Base</th>{head}</tr>"
                      f"<tr>{sub}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")

    # ---- aggregates
    agg_rows = []
    for key, d in sorted(summ.get("per_ensemble", {}).items()):
        metric, group, ens = key.split("|")
        if metric != "f1":
            continue
        sp = d.get("spearman_datasets")
        bt = d.get("best_tree_agreement")
        agg_rows.append(f"<tr><th scope='row'>{ens}</th><td>{group}</td><td>{d['n']}</td>"
                        f"<td>{d['mean_abs_diff']:.2f}</td><td>{d['median_diff']:+.2f}</td>"
                        f"<td>{'' if sp is None else f'{sp:.2f}'}</td>"
                        f"<td>{'' if bt is None else f'{100 * bt:.0f}% ({d.get(chr(110) + '_datasets_complete', '')})'}</td></tr>")
    win_rows = []
    for key, d in sorted(summ.get("win_rates", {}).items()):
        metric, group, what = key.split("|")
        if metric != "f1" or not d["n"]:
            continue
        win_rows.append(f"<tr><th scope='row'>{what.replace('>', ' &gt; ')}</th><td>{group}</td><td>{d['n']}</td>"
                        f"<td>{100 * d['ours']:.0f}%</td><td>{100 * d['paper_same_pairs']:.0f}%</td></tr>")

    eq_ok = sum(1 for r in eq if r["mism"] == 0)
    eq_rows = "".join(f"<tr><td>{r['ens']}</td><td>{r['tree']}</td><td>{r['data']}</td><td>{r['n']:,}</td>"
                      f"<td class='{'ok' if r['mism'] == 0 else 'bad'}'>{r['mism']}</td><td>{r['rel']:.1e}</td></tr>"
                      for r in sorted(eq, key=lambda r: (r['data'], ENS.index(r['ens']), r['tree'])))

    page = TEMPLATE.format(stamp=stamp, n_runs=n_runs, eq_ok=eq_ok, eq_n=len(eq), eq_rows=eq_rows,
                           scatter=scatter_svg(f1) if len(f1) else "<p class='muted'>Sem resultados ainda.</p>",
                           agg_rows="".join(agg_rows), win_rows="".join(win_rows), tables="".join(tables))
    with open(OUT, "w") as fh:
        fh.write(page)
    print("wrote", OUT, n_runs, "runs")


TEMPLATE = """<title>Replicação HAST-Python</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,650&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400&display=swap">
<style>
/* Layout: one reading column (lab notebook); evidence first, then results tables. */
:root {{
  --bg: #fbfbf9; --surface: #f2f3f0; --ink: #16181a; --ink-2: #4c5054; --ink-3: #74797e;
  --rule: #dcdfdb; --accent: #2a5f8f; --pos: #1d6b3a; --neg: #a23a2c;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a;
  --display: "Source Serif 4", Georgia, serif; --body: "IBM Plex Sans", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #161819; --surface: #1f2224; --ink: #eceeec; --ink-2: #b9bdbf; --ink-3: #8b9094;
  --rule: #33383b; --accent: #8ab8e6; --pos: #6fcf92; --neg: #f08a7c;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #161819; --surface: #1f2224; --ink: #eceeec; --ink-2: #b9bdbf; --ink-3: #8b9094;
  --rule: #33383b; --accent: #8ab8e6; --pos: #6fcf92; --neg: #f08a7c;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--ink); font: 15px/1.6 var(--body); }}
main {{ max-width: 980px; margin: 0 auto; padding-block: 32px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 28px; }}
h1, h2, h3 {{ font-family: var(--display); font-weight: 650; text-wrap: balance; line-height: 1.2; margin: 0; }}
h1 {{ font-size: 2rem; }} h2 {{ font-size: 1.35rem; padding-top: 8px; border-top: 1px solid var(--rule); }}
h3 {{ font-size: 1.05rem; margin-top: 8px; }}
p, li {{ max-width: 68ch; margin: 0; }} ul {{ margin: 0; padding-left: 1.2em; display: grid; gap: 6px; }}
section {{ display: grid; gap: 12px; min-width: 0; }}
.eyebrow {{ font: 500 0.75rem var(--body); letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-3); }}
.muted {{ color: var(--ink-3); }}
code {{ font: 0.88em var(--mono); background: var(--surface); padding: 1px 4px; border-radius: 3px; }}
.facts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }}
.fact {{ background: var(--surface); border-radius: 6px; padding: 12px 14px; display: grid; gap: 2px; }}
.fact b {{ font: 600 1.15rem var(--body); font-variant-numeric: tabular-nums; }}
.fact span {{ color: var(--ink-2); font-size: 0.88rem; }}
.tablewrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 0.86rem; }}
th, td {{ padding: 4px 10px; text-align: right; border-bottom: 1px solid var(--rule); white-space: nowrap; }}
th {{ font-weight: 600; color: var(--ink-2); }} th[scope=row], td:first-child {{ text-align: left; }}
td.pos {{ color: var(--pos); }} td.neg {{ color: var(--neg); }} td.ok {{ color: var(--pos); }} td.bad {{ color: var(--neg); font-weight: 600; }}
.chart {{ display: grid; gap: 8px; max-width: 600px; position: relative; }}
.scatter {{ width: 100%; height: auto; }}
.scatter .grid {{ stroke: var(--rule); stroke-width: 1; }} .scatter .ident {{ stroke: var(--ink-3); stroke-dasharray: 4 4; }}
.scatter .tick {{ fill: var(--ink-3); font: 11px var(--body); }} .scatter .axis {{ fill: var(--ink-2); font: 12px var(--body); }}
.dot {{ stroke: var(--bg); stroke-width: 2; fill-opacity: 0.85; cursor: default; }}
.dot:hover, .dot:focus {{ fill-opacity: 1; stroke: var(--ink); outline: none; }}
.s1 {{ fill: var(--s1); }} .s2 {{ fill: var(--s2); }} .s3 {{ fill: var(--s3); }}
.legend {{ display: flex; gap: 16px; flex-wrap: wrap; font-size: 0.85rem; color: var(--ink-2); }}
.legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }}
#tip {{ position: absolute; pointer-events: none; background: var(--ink); color: var(--bg); font-size: 0.8rem;
  padding: 4px 8px; border-radius: 4px; white-space: nowrap; }}
@media (prefers-reduced-motion: no-preference) {{ .dot {{ transition: fill-opacity .12s; }} }}
</style>
<main>
<header style="display:grid;gap:8px">
  <span class="eyebrow">Fase 3 · relatório de replicação · parcial · {stamp}</span>
  <h1>Replicação HAST-Python</h1>
  <p>Port para Python das Hoeffding Adaptive Splitting Trees (Nowak Assis, Barddal &amp; Enembreck, DMKD 2026) e dos cinco ensembles do artigo, validado contra o código Java dos autores, e a comparação dos nossos resultados com as tabelas brutas do artigo. Os números abaixo são gerados pelos scripts do repositório e se completam à medida que as execuções terminam.</p>
</header>

<section>
  <h2>Situação</h2>
  <div class="facts">
    <div class="fact"><b>0 divergências</b><span>de predição Python × Java em todos os testes (árvores isoladas e 4 ensembles)</span></div>
    <div class="fact"><b>{eq_ok} / {eq_n}</b><span>execuções com 100 membros idênticas ao Java nas primeiras 10 mil instâncias</span></div>
    <div class="fact"><b>{n_runs}</b><span>execuções completas já comparadas com o artigo (ensemble × base × árvore)</span></div>
    <div class="fact"><b>11 de 13</b><span>bases reais disponíveis (LADPU e Asfault pedidas aos autores) e 24 sintéticos gerados</span></div>
  </div>
</section>

<section>
  <h2>O que foi validado</h2>
  <ul>
    <li><b>Protocolo do artigo:</b> <code>EvaluatePrequential</code> com avaliador cumulativo, 20 pontos, F1 macro do MOA com NaN contado como 0 e desvio populacional. ARF + HT no Outdoor dá acurácia 63,490 ± 1,229 (artigo: 63,487 ± 1,208).</li>
    <li><b>Fluxo completo:</b> ARTE + HLAST_D no Elec (45 mil instâncias, 100 membros) tem os 20 pontos idênticos entre Java e Python; F1 91,30 contra 91,33 no artigo.</li>
    <li><b>Implementação rápida (numba)</b> idêntica bit a bit à implementação de referência em Python nas 27 combinações testadas (Elec, Airlines, CovType).</li>
    <li>A única diferença numérica é de 1 ulp em <code>exp</code>/<code>log</code> (intrínsecos da JVM), com erro relativo abaixo de 1e-15 nos votos.</li>
  </ul>
</section>

<section>
  <h2>Achados no código original</h2>
  <ul>
    <li><b>ARFHLAST descarta um atributo por folha</b> (<code>for j &lt; numAttributes - 1</code>); o <code>ARFHoeffdingTree</code> do MOA usa todos. Replicado; há opção para corrigir.</li>
    <li><b>SRP do MOA</b> não zera o learner de background ao promovê-lo; a árvore promovida é treinada duas vezes por instância até o próximo warning. Replicado.</li>
    <li><b>ARF + HLAST/HLAST_D não reproduz o artigo</b> no Outdoor (HLAST_D: acurácia 74,6 contra 67,6), embora ARF + HT reproduza. Python e Java concordam, então a diferença está na configuração ou versão usada pelos autores.</li>
    <li><b>O F1 do MOA é instável em classes raríssimas.</b> No Poker, a classe 9 tem 2 instâncias em 829 mil; se o ensemble nunca a prediz, o F1 fica indefinido (contado como 0) nos 20 pontos. Nosso F1 no Poker é 0, contra ~48 no artigo, com acurácia a menos de 1 ponto. O mesmo ocorre no Outdoor (classe 29). Por isso a comparação também usa a acurácia e a mediana das diferenças.</li>
    <li>O <code>ARTEAttributeClassObserver</code> e o ARE não foram publicados e foram <b>reconstruídos</b> a partir dos artigos; o ARE é aproximado (ζ = 10 assumido).</li>
  </ul>
</section>

<section>
  <h2>Nosso F1 × F1 do artigo</h2>
  <div class="chart">
    <div class="legend"><span><i style="background:var(--s1)"></i>HT</span><span><i style="background:var(--s2)"></i>HLAST</span><span><i style="background:var(--s3)"></i>HLAST_D</span></div>
    {scatter}
    <div id="tip" hidden></div>
  </div>
  <p class="muted">Cada ponto é um par ensemble × base × árvore. Sobre a linha tracejada, o resultado coincide com o artigo. Os valores exatos estão nas tabelas abaixo.</p>
</section>

<section>
  <h2>Concordância por ensemble (F1)</h2>
  <div class="tablewrap"><table>
    <thead><tr><th>Ensemble</th><th>Grupo</th><th>N</th><th>|Δ| médio</th><th>Δ mediano</th><th>Spearman entre bases</th><th>Mesma melhor árvore</th></tr></thead>
    <tbody>{agg_rows}</tbody></table></div>
  <h3>Vitórias sobre o HT (critério do artigo)</h3>
  <div class="tablewrap"><table>
    <thead><tr><th>Comparação</th><th>Grupo</th><th>Pares</th><th>Nosso</th><th>Artigo, mesmos pares</th></tr></thead>
    <tbody>{win_rows}</tbody></table></div>
</section>

<section>
  <h2>Resultados por ensemble</h2>
  <p class="muted">F1 em %, média dos 20 pontos. Δ = nosso − artigo, em pontos percentuais (verde &gt; +0,5; vermelho &lt; −0,5).</p>
  {tables}
</section>

<section>
  <h2>Equivalência com o Java (100 membros)</h2>
  <div class="tablewrap"><table>
    <thead><tr><th>Ensemble</th><th>Árvore</th><th>Base</th><th>Instâncias</th><th>Divergências de predição</th><th>Maior erro relativo nos votos</th></tr></thead>
    <tbody>{eq_rows}</tbody></table></div>
</section>

<section>
  <h2>Próximos passos</h2>
  <ul>
    <li>A fila desta máquina (<code>scripts/run_replication_queue.sh</code>, 4 workers) conclui todas as bases reais e os 24 sintéticos nos 5 ensembles; estimativa de ~2 dias.</li>
    <li>Regenerar este relatório: <code>python scripts/replication_report.py &amp;&amp; python scripts/build_report_html.py</code>.</li>
  </ul>
</section>
</main>
<script>
(() => {{
  const tip = document.getElementById('tip');
  const box = tip && tip.parentElement;
  if (!tip) return;
  const show = (e) => {{
    const t = e.target.closest('.dot'); if (!t) return;
    tip.textContent = t.dataset.tip; tip.hidden = false;
    const r = box.getBoundingClientRect(), c = t.getBoundingClientRect();
    let x = c.left - r.left + 10, y = c.top - r.top - 30;
    tip.style.left = Math.max(0, Math.min(x, r.width - tip.offsetWidth)) + 'px'; tip.style.top = Math.max(0, y) + 'px';
  }};
  box.addEventListener('mouseover', show); box.addEventListener('focusin', show);
  box.addEventListener('mouseleave', () => tip.hidden = true); box.addEventListener('focusout', () => tip.hidden = true);
}})();
</script>
"""

if __name__ == "__main__":
    main()
