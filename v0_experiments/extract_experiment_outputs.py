#!/usr/bin/env python3
"""Extrai as saidas das celulas de experimento dos notebooks e grava um JSON por experimento."""
import json, re, os, glob, sys
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")

# ------------------------------------------------------------------ helpers
def cell_text(cell):
    """Concatena todo o stdout/stderr de uma celula."""
    parts = []
    for o in cell.get("outputs", []):
        if o["output_type"] == "stream":
            parts.append("".join(o["text"]))
    return "".join(parts)

def cell_richtext(cell):
    parts = []
    for o in cell.get("outputs", []):
        if o["output_type"] == "stream":
            parts.append("".join(o["text"]))
        elif "data" in o and "text/plain" in o["data"]:
            parts.append("".join(o["data"]["text/plain"]))
    return "".join(parts)

def has_image(cell):
    return any("data" in o and "image/png" in o["data"] for o in cell.get("outputs", []))

def slug(s):
    s = re.sub(r"[^\w\-.]+", "_", s.strip())
    return re.sub(r"_+", "_", s).strip("_")

def num(x):
    try:
        f = float(x)
        return int(f) if f.is_integer() and abs(f) < 1e15 and "." not in x and "e" not in x.lower() else f
    except Exception:
        return x

# --------------------------------------------------- parse do codigo-fonte
def balanced_call(src, fname):
    """Devolve o texto dos argumentos da primeira chamada fname(...) balanceada."""
    i = src.find(fname + "(")
    if i < 0:
        return None
    j = i + len(fname)
    depth = 0
    for k in range(j, len(src)):
        if src[k] in "([{":
            depth += 1
        elif src[k] in ")]}":
            depth -= 1
            if depth == 0:
                return src[j + 1:k]
    return None

SIMPLE_KW = re.compile(r"(?m)^\s{0,12}(\w+)\s*=\s*([^,\n]+?)\s*,?\s*(?:#.*)?$")

def parse_kwargs(argtext):
    """Extrai kwargs simples (nivel superior) do texto de argumentos."""
    out = OrderedDict()
    if not argtext:
        return out
    # remove comentarios
    lines = [re.sub(r"#.*$", "", ln) for ln in argtext.split("\n")]
    depth = 0
    buf = []
    chunks = []
    for ln in lines:
        for ch in ln:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            if ch == "," and depth == 0:
                chunks.append("".join(buf)); buf = []
            else:
                buf.append(ch)
        buf.append("\n")
    chunks.append("".join(buf))
    for ch in chunks:
        m = re.match(r"\s*(\w+)\s*=\s*(.+)\s*$", ch.strip(), re.S)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if val.startswith(("'", '"')) and val.endswith(("'", '"')) and len(val) > 1:
            out[key] = val[1:-1]
        elif re.fullmatch(r"[-+]?\d+", val):
            out[key] = int(val)
        elif re.fullmatch(r"[-+]?\d*\.?\d+([eE][-+]?\d+)?", val):
            out[key] = float(val)
        elif val in ("True", "False"):
            out[key] = val == "True"
        elif val == "None":
            out[key] = None
        elif val.startswith("f\"") or val.startswith("f'"):
            out[key] = val[2:-1]          # f-string -> template cru
        else:
            out[key] = val                # expressao / referencia a variavel
    return out

MODEL_RE = re.compile(r"(?m)^\s*(?:base_)?model(?:_base)?\s*=\s*([A-Za-z_]\w*)\s*\(")

# -------------------------------------------------------- parse das saidas
RE_ROUND_START = re.compile(r">>> Iniciando Round (\d+)/(\d+) <<<")
RE_OUTER_FOLD  = re.compile(r"=== Outer Fold (\d+)/(\d+) ===")
RE_FOLD_TRAIN  = re.compile(r"\[Fold (\d+)\] (Classifier|AutoEncoder) training \((\d+) epochs\)")
RE_PRETRAIN    = re.compile(r"\[Pre-train\] Epoch (\d+)/(\d+) Recon Loss: ([\d.eE+-]+)")
RE_SUPERVISED  = re.compile(r"\[Supervised\] Epoch (\d+)/(\d+) Train Loss: ([\d.eE+-]+), Val Loss: ([\d.eE+-]+), Time: ([\d.]+)s")
RE_FOLD_RESULT = re.compile(r"Result: Acc=([\d.]+), F1=([\d.]+)")
RE_MEAN_ACC    = re.compile(r"^Mean Accuracy: ([\d.]+)", re.M)
RE_ROUND_DONE  = re.compile(r"Round (\d+) Finalizado: Acc=([\d.]+)")
RE_INPUT_LEN   = re.compile(r"Input length: (\d+)")
RE_NUM_CLASSES = re.compile(r"Num classes: (\d+)")
RE_TOTAL_ROUNDS= re.compile(r"Total de Rounds(?: para executar)?: (\d+)")
RE_ROUNDS_EXEC = re.compile(r"(?:Total Rounds Executados|Rounds Executados): (\d+)")
RE_ACC_GLOBAL  = re.compile(r"Acur[áa]cia M[ée]dia(?: Global)?:\s*([\d.]+)\s*±\s*([\d.]+)")
RE_F1_GLOBAL   = re.compile(r"F1-Score M[ée]dio(?: Global)?:\s*([\d.]+)\s*±\s*([\d.]+)")

def parse_fold_block(text, fold_no, total_folds):
    fold = OrderedDict()
    fold["fold"] = fold_no
    fold["total_folds"] = total_folds
    m = RE_FOLD_TRAIN.search(text)
    if m:
        fold["fold_index"] = int(m.group(1))
        fold["training_stage"] = "autoencoder+classifier" if m.group(2) == "AutoEncoder" else "classifier"
        fold["epochs_logged_for"] = int(m.group(3))
    pre = [OrderedDict(epoch=int(a), total_epochs=int(b), recon_loss=float(c))
           for a, b, c in RE_PRETRAIN.findall(text)]
    sup = [OrderedDict(epoch=int(a), total_epochs=int(b), train_loss=float(c),
                       val_loss=float(d), time_seconds=float(e))
           for a, b, c, d, e in RE_SUPERVISED.findall(text)]
    r = RE_FOLD_RESULT.search(text)
    fold["accuracy"] = float(r.group(1)) if r else None
    fold["f1_score"] = float(r.group(2)) if r else None
    hist = OrderedDict()
    if pre:
        hist["pretrain"] = pre
    if sup:
        hist["supervised"] = sup
    fold["training_history"] = hist
    return fold

def split_folds(text):
    """Divide um bloco em folds a partir dos marcadores '=== Outer Fold i/n ==='."""
    marks = list(RE_OUTER_FOLD.finditer(text))
    folds = []
    for idx, m in enumerate(marks):
        end = marks[idx + 1].start() if idx + 1 < len(marks) else len(text)
        folds.append(parse_fold_block(text[m.end():end], int(m.group(1)), int(m.group(2))))
    return folds

def parse_run_output(text):
    """Interpreta a saida da celula que executa o experimento."""
    res = OrderedDict()
    m = RE_INPUT_LEN.search(text)
    if m:
        res["input_length"] = int(m.group(1))
    m = RE_NUM_CLASSES.search(text)
    if m:
        res["num_classes"] = int(m.group(1))

    round_marks = list(RE_ROUND_START.finditer(text))
    round_done = {int(a): float(b) for a, b in RE_ROUND_DONE.findall(text)}

    if round_marks:
        res["protocol"] = "multi_round"
        m = RE_TOTAL_ROUNDS.search(text)
        if m:
            res["total_rounds_planned"] = int(m.group(1))
        rounds = []
        for idx, rm in enumerate(round_marks):
            end = round_marks[idx + 1].start() if idx + 1 < len(round_marks) else len(text)
            block = text[rm.end():end]
            rno = int(rm.group(1))
            r = OrderedDict()
            r["round"] = rno
            r["total_rounds"] = int(rm.group(2))
            r["folds"] = split_folds(block)
            ma = RE_MEAN_ACC.search(block)
            r["mean_accuracy"] = float(ma.group(1)) if ma else None
            if rno in round_done:
                r["reported_accuracy"] = round_done[rno]
            accs = [f["accuracy"] for f in r["folds"] if f["accuracy"] is not None]
            f1s = [f["f1_score"] for f in r["folds"] if f["f1_score"] is not None]
            r["fold_accuracy_mean"] = round(sum(accs) / len(accs), 6) if accs else None
            r["fold_f1_mean"] = round(sum(f1s) / len(f1s), 6) if f1s else None
            rounds.append(r)
        res["rounds"] = rounds
        summary = OrderedDict()
        m = RE_ROUNDS_EXEC.search(text)
        if m:
            summary["rounds_executed"] = int(m.group(1))
        m = RE_ACC_GLOBAL.search(text)
        if m:
            summary["mean_accuracy"] = float(m.group(1))
            summary["std_accuracy"] = float(m.group(2))
        m = RE_F1_GLOBAL.search(text)
        if m:
            summary["mean_f1_score"] = float(m.group(1))
            summary["std_f1_score"] = float(m.group(2))
        if summary:
            res["global_summary"] = summary
    else:
        res["protocol"] = "single_round"
        res["folds"] = split_folds(text)
        ma = RE_MEAN_ACC.search(text)
        if ma:
            res["mean_accuracy"] = float(ma.group(1))
    return res

# ------------------------------------------------ parse do show_results()
def parse_show_results(text):
    if "EXPERIMENTO:" not in text:
        return None
    out = OrderedDict()
    for key, field in (("EXPERIMENTO", "experiment_name"), ("DESCRIÇÃO", "description"),
                       ("MODELO", "model")):
        m = re.search(key + r":\s*(.+)", text)
        if m:
            out[field] = m.group(1).strip()
    m = re.search(r"QUANTIDADE DE FOLDS:\s*(\d+)", text)
    if m:
        out["num_folds"] = int(m.group(1))

    metrics = OrderedDict()
    for pat, field in ((r"Acur[áa]cia M[ée]dia\s+([\d.]+)", "mean_accuracy"),
                       (r"Desvio Padr[ãa]o Acur[áa]cia\s+±([\d.]+)", "std_accuracy"),
                       (r"F1-Score M[ée]dio\s+([\d.]+)", "mean_f1_score"),
                       (r"Desvio Padr[ãa]o F1-Score\s+±([\d.]+)", "std_f1_score")):
        m = re.search(pat, text)
        if m:
            metrics[field] = float(m.group(1))
    out["overall_metrics"] = metrics

    folds = []
    tail = text.split("DETALHES POR FOLD:", 1)
    if len(tail) == 2:
        for ln in tail[1].split("\n"):
            m = re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", ln)
            if m:
                folds.append(OrderedDict(fold=int(m.group(1)), accuracy=float(m.group(2)),
                                         f1_score=float(m.group(3)), precision=float(m.group(4)),
                                         recall=float(m.group(5))))
    out["per_fold_metrics"] = folds

    cm = OrderedDict()
    m = re.search(r"Acur[áa]cia Geral:\s*([\d.]+)", text)
    if m:
        cm["overall_accuracy"] = float(m.group(1))
    m = re.search(r"Total de Amostras:\s*(\d+)", text)
    if m:
        cm["total_samples"] = int(m.group(1))
    if cm:
        out["confusion_matrix_stats"] = cm
    return out

# ------------------------------------------------- info de folds/rounds
RE_FOLD_COMP = re.compile(r"fold:\s*(\d+)\s*->\s*(.*?)\s*=>\s*(\d+)")

def parse_fold_design(nb):
    """Le a celula geradora de folds (composicao por round/fold)."""
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        txt = cell_richtext(cell)
        if "fold: " not in txt:
            continue
        design = OrderedDict()
        m = re.search(r"Total combs:\s*(\d+)", txt)
        if m:
            design["total_combinations"] = int(m.group(1))
        rounds = []
        cur = None
        for ln in txt.split("\n"):
            mr = re.match(r"round:\s*(\d+)", ln)
            if mr:
                cur = OrderedDict(round=int(mr.group(1)), folds=[])
                rounds.append(cur)
                continue
            mf = RE_FOLD_COMP.search(ln)
            if mf and cur is not None:
                groups = [g.strip() for g in mf.group(2).split(",") if g.strip()]
                cur["folds"].append(OrderedDict(fold=int(mf.group(1)),
                                                groups=groups,
                                                combination_id=int(mf.group(3))))
        design["rounds"] = rounds
        return design
    return None

# ------------------------------------------------------------- pipeline
def preceding_heading(cells, i):
    for j in range(i - 1, -1, -1):
        if cells[j]["cell_type"] == "markdown":
            src = "".join(cells[j]["source"]).strip()
            if src.startswith("#"):
                return src.lstrip("#").strip()
    return None

def process_notebook(path, suite):
    nb = json.load(open(path, encoding="utf-8"))
    cells = nb["cells"]
    dataset = re.sub(r"^SignalAI_Framework_", "", os.path.basename(path)[:-6])
    fold_design = parse_fold_design(nb)

    experiments = []
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "DeepLearningExperiment(" not in src or ".run()" not in src:
            continue

        args = parse_kwargs(balanced_call(src, "DeepLearningExperiment"))
        raw_name = str(args.get("name", ""))
        name_template = raw_name if "{" in raw_name else None
        base_name = re.sub(r"_?round_?\{[^}]*\}", "", raw_name).strip("_") or None
        desc = args.get("description")
        if isinstance(desc, str) and "{" in desc:
            desc_template, desc = desc, re.sub(r"\s*\(Round \{[^}]*\}\)", "", desc)
        else:
            desc_template = None
        mm = MODEL_RE.search(src)
        model = mm.group(1) if mm else None

        run_out = cell_text(cell)
        # celula seguinte com show_results()
        show, show_idx, cm_img = None, None, False
        for j in (i + 1, i + 2):
            if j < len(cells) and cells[j]["cell_type"] == "code" and "show_results" in "".join(cells[j]["source"]):
                show = parse_show_results(cell_text(cells[j]))
                show_idx = j
                cm_img = has_image(cells[j])
                break

        parsed = parse_run_output(run_out) if run_out.strip() else {}
        executed = bool(parsed.get("folds") or parsed.get("rounds")) or bool(show)

        exp = OrderedDict()
        exp["experiment_name"] = (show or {}).get("experiment_name") or base_name or f"experiment_cell_{i}"
        if name_template:
            exp["experiment_name_template"] = name_template
        exp["dataset"] = dataset
        exp["suite"] = suite
        exp["protocol"] = parsed.get("protocol", "multi_round" if name_template else "single_round")
        exp["model"] = model or (show or {}).get("model")
        exp["description"] = (show or {}).get("description") or desc
        if desc_template:
            exp["description_template"] = desc_template
        exp["section"] = preceding_heading(cells, i)
        exp["status"] = "executed" if executed else "not_executed"
        exp["source"] = OrderedDict(
            notebook=os.path.relpath(path, ROOT),
            run_cell_index=i,
            results_cell_index=show_idx,
        )

        cfg = OrderedDict()
        for k in ("batch_size", "lr", "num_epochs", "pretrain_epochs", "recon_loss_weight",
                  "criterion", "reconstruction_criterion", "output_dir", "data_fold_idxs"):
            if k in args:
                cfg[k] = args[k]
        for k, v in args.items():
            if k not in cfg and k not in ("name", "description", "dataset", "model"):
                cfg[k] = v
        if "input_length" in parsed:
            cfg["input_length"] = parsed["input_length"]
        if "num_classes" in parsed:
            cfg["num_classes"] = parsed["num_classes"]
        exp["configuration"] = cfg

        results = OrderedDict()
        if show:
            results["summary"] = show["overall_metrics"]
            if show.get("num_folds") is not None:
                results["summary"]["num_folds"] = show["num_folds"]
            results["per_fold_metrics"] = show["per_fold_metrics"]
            if "confusion_matrix_stats" in show:
                results["confusion_matrix"] = show["confusion_matrix_stats"]
                results["confusion_matrix"]["figure_in_notebook"] = cm_img
        if parsed.get("protocol") == "multi_round":
            if "total_rounds_planned" in parsed:
                results["total_rounds_planned"] = parsed["total_rounds_planned"]
            if "global_summary" in parsed:
                results["summary"] = parsed["global_summary"]
            results["rounds"] = parsed["rounds"]
        elif parsed.get("folds") is not None:
            if "mean_accuracy" in parsed:
                results.setdefault("summary", OrderedDict())
                results["summary"].setdefault("mean_accuracy", parsed["mean_accuracy"])
            results["folds_training_log"] = parsed["folds"]
        exp["results"] = results

        if executed and exp["protocol"] != suite:
            exp["notes"] = ("A celula esta no conjunto '%s_experiments', mas a saida registrada "
                            "e de uma execucao %s (a celula usa 'folds_singleround_deep' e a saida "
                            "nao contem rodadas)." % (suite, exp["protocol"]))
        if fold_design and exp["protocol"] == "multi_round":
            exp["fold_design"] = fold_design
        experiments.append(exp)
    return dataset, experiments, fold_design

def main():
    index_all = []
    for suite in ("single_round", "multi_round"):
        for path in sorted(glob.glob(os.path.join(ROOT, suite + "_experiments", "*.ipynb"))):
            dataset, exps, fold_design = process_notebook(path, suite)
            outdir = os.path.join(OUT, suite, dataset)
            os.makedirs(outdir, exist_ok=True)
            seen = {}
            nb_index = []
            for exp in exps:
                base = slug(exp["experiment_name"])
                seen[base] = seen.get(base, 0) + 1
                fname = base if seen[base] == 1 else f"{base}_{seen[base]}"
                fpath = os.path.join(outdir, fname + ".json")
                with open(fpath, "w", encoding="utf-8") as fh:
                    json.dump(exp, fh, ensure_ascii=False, indent=2)
                    fh.write("\n")
                s = exp["results"].get("summary", {})
                nb_index.append(OrderedDict(
                    file=fname + ".json",
                    experiment_name=exp["experiment_name"],
                    model=exp["model"],
                    protocol=exp["protocol"],
                    status=exp["status"],
                    mean_accuracy=s.get("mean_accuracy"),
                    std_accuracy=s.get("std_accuracy"),
                    mean_f1_score=s.get("mean_f1_score"),
                    std_f1_score=s.get("std_f1_score"),
                ))
                index_all.append(OrderedDict(suite=suite, dataset=dataset, **nb_index[-1]))
            idx = OrderedDict(dataset=dataset, suite=suite,
                              source_notebook=os.path.relpath(path, ROOT),
                              num_experiments=len(exps))
            if fold_design:
                idx["fold_design"] = fold_design
            idx["experiments"] = nb_index
            with open(os.path.join(outdir, "_index.json"), "w", encoding="utf-8") as fh:
                json.dump(idx, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            print(f"{suite}/{dataset}: {len(exps)} experimentos "
                  f"({sum(1 for e in exps if e['status']=='executed')} executados)")
    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(OrderedDict(total_experiments=len(index_all), experiments=index_all),
                  fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print("total:", len(index_all))

if __name__ == "__main__":
    main()
