import argparse
import math
import os
import re
import sys
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

STOPWORDS = {
    "a","an","and","are","as","at","be","but","by","for","if","in","into","is","it","no",
    "not","of","on","or","such","that","the","their","then","there","these","they","this",
    "to","was","will","with","from","we","you","your","our","were","has","have","had"
}

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

def tokenize(text: str, lowercase: bool = True, remove_stop: bool = True) -> List[str]:
    toks = TOKEN_RE.findall(text)
    if lowercase:
        toks = [t.lower() for t in toks]
    if remove_stop:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks

@dataclass
class Document:
    doc_id: int
    name: str
    text: str
    tokens: List[str]

def load_corpus_from_folder(folder: str) -> List[Document]:
    docs: List[Document] = []
    did = 0
    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)
        if os.path.isfile(path) and fname.lower().endswith((".txt", ".md")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            docs.append(Document(did, fname, txt, tokenize(txt)))
            did += 1
    if not docs:
        raise RuntimeError(f"No .txt or .md files found in: {folder}")
    return docs

def demo_corpus() -> List[Document]:
    raw = {
        "doc1.txt": "Information retrieval is the science of searching for information in documents.",
        "doc2.txt": "The vector space model represents documents and queries as vectors.",
        "doc3.txt": "Boolean retrieval uses set operations like AND, OR, and NOT.",
        "doc4.txt": "The probabilistic model ranks documents by estimating the probability of relevance.",
        "doc5.txt": "BM25 is a popular probabilistic retrieval function used in modern search engines.",
        "doc6.txt": "Term frequency and inverse document frequency (TF-IDF) are core to vector space models."
    }
    docs = []
    for i, (name, txt) in enumerate(sorted(raw.items())):
        docs.append(Document(i, name, txt, tokenize(txt)))
    return docs

class BooleanModel:
    def __init__(self, docs: List[Document]):
        self.N = len(docs)
        self.docs = docs
        self.index: Dict[str, Set[int]] = defaultdict(set)
        for d in docs:
            for t in set(d.tokens):
                self.index[t].add(d.doc_id)
        self.universe: Set[int] = {d.doc_id for d in docs}

    def _to_rpn(self, query_tokens: List[str]) -> List[str]:
        prec = {"NOT": 3, "AND": 2, "OR": 1}
        output = []
        stack: List[str] = []
        for tok in query_tokens:
            u = tok.upper()
            if u in ("AND","OR","NOT"):
                while stack and stack[-1] in prec and (
                    (prec[stack[-1]] > prec[u]) or (prec[stack[-1]] == prec[u] and u != "NOT")
                ):
                    output.append(stack.pop())
                stack.append(u)
            elif tok == "(":
                stack.append(tok)
            elif tok == ")":
                while stack and stack[-1] != "(":
                    output.append(stack.pop())
                if not stack:
                    raise ValueError("Mismatched parentheses")
                stack.pop()
            else:
                output.append(tok)
        while stack:
            op = stack.pop()
            if op in ("(", ")"):
                raise ValueError("Mismatched parentheses")
            output.append(op)
        return output

    def _tokenize_query(self, q: str) -> List[str]:
        parts = re.findall(r"\(|\)|AND|OR|NOT|[A-Za-z0-9]+", q, flags=re.IGNORECASE)
        normalized = []
        for p in parts:
            if re.fullmatch(r"[A-Za-z0-9]+", p):
                toks = tokenize(p, lowercase=True, remove_stop=True)
                if toks:
                    normalized.append(toks[0])
            else:
                normalized.append(p.upper())
        return normalized

    def _eval_rpn(self, rpn: List[str]) -> Set[int]:
        st: List[Set[int]] = []
        for tok in rpn:
            U = tok.upper()
            if U == "NOT":
                a = st.pop() if st else set()
                st.append(self.universe - a)
            elif U == "AND":
                b = st.pop() if st else set()
                a = st.pop() if st else set()
                st.append(a & b)
            elif U == "OR":
                b = st.pop() if st else set()
                a = st.pop() if st else set()
                st.append(a | b)
            else:
                st.append(self.index.get(tok, set()))
        return st.pop() if st else set()

    def query(self, q: str) -> List[Tuple[int, float]]:
        toks = self._tokenize_query(q)
        rpn = self._to_rpn(toks)
        result_set = self._eval_rpn(rpn)
        return [(doc_id, 1.0) for doc_id in sorted(result_set)]

class VectorSpaceModel:
    def __init__(self, docs: List[Document]):
        self.docs = docs
        self.N = len(docs)
        self.df: Dict[str, int] = defaultdict(int)
        for d in docs:
            for t in set(d.tokens):
                self.df[t] += 1
        self.idf: Dict[str, float] = {t: math.log((self.N + 1) / (df + 1)) + 1.0 for t, df in self.df.items()}
        self.doc_tf: List[Counter] = [Counter(d.tokens) for d in docs]
        self.doc_vecs: List[Dict[str, float]] = []
        self.doc_norm: List[float] = []
        for tf in self.doc_tf:
            vec = {t: (tf[t]) * self.idf.get(t, 0.0) for t in tf}
            norm = math.sqrt(sum(w*w for w in vec.values())) or 1.0
            self.doc_vecs.append(vec)
            self.doc_norm.append(norm)

    def _query_vec(self, q: str) -> Tuple[Dict[str, float], float]:
        qtoks = tokenize(q)
        qtf = Counter(qtoks)
        qvec = {t: qtf[t] * self.idf.get(t, math.log((self.N + 1) / 1) + 1.0) for t in qtf}
        qnorm = math.sqrt(sum(w*w for w in qvec.values())) or 1.0
        return qvec, qnorm

    def query(self, q: str, top_k: int = 10) -> List[Tuple[int, float]]:
        qvec, qnorm = self._query_vec(q)
        scores: List[Tuple[int, float]] = []
        for i, dvec in enumerate(self.doc_vecs):
            dot = 0.0
            (small, big) = (qvec, dvec) if len(qvec) <= len(dvec) else (dvec, qvec)
            for t, w in small.items():
                if t in big:
                    dot += w * big[t]
            sim = dot / (qnorm * self.doc_norm[i])
            if sim > 0:
                scores.append((i, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

class BM25:
    def __init__(self, docs: List[Document], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.N = len(docs)
        self.k1 = k1
        self.b = b
        self.doc_len = [len(d.tokens) for d in docs]
        self.avgdl = sum(self.doc_len) / max(1, self.N)
        self.df: Dict[str, int] = defaultdict(int)
        for d in docs:
            for t in set(d.tokens):
                self.df[t] += 1
        self.idf: Dict[str, float] = {
            t: math.log((self.N - df + 0.5) / (df + 0.5) + 1e-10) for t, df in self.df.items()
        }
        self.tf_list: List[Counter] = [Counter(d.tokens) for d in docs]

    def _score_doc(self, i: int, qterms: List[str]) -> float:
        score = 0.0
        dl = self.doc_len[i]
        tf = self.tf_list[i]
        for t in qterms:
            if t not in self.idf:
                continue
            f = tf.get(t, 0)
            if f == 0:
                continue
            idf = self.idf[t]
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (f * (self.k1 + 1)) / denom
        return score

    def query(self, q: str, top_k: int = 10) -> List[Tuple[int, float]]:
        qterms = tokenize(q)
        scores = []
        for i in range(self.N):
            s = self._score_doc(i, qterms)
            if s != 0.0:
                scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

def print_results(model_name: str, docs: List[Document], results: List[Tuple[int, float]], limit: int = 10):
    print(f"\n=== {model_name} Results (top {min(limit, len(results))}) ===")
    for rank, (doc_id, score) in enumerate(results[:limit], start=1):
        name = docs[doc_id].name
        snippet = " ".join(docs[doc_id].text.split()[:18])
        print(f"{rank:2d}. Doc#{doc_id:02d} [{name}]  Score={score:.6f}  :: {snippet}...")

def save_results_csv(filename, docs, results_dict):
    max_len = max(len(r) for r in results_dict.values())
    header = []
    for model in results_dict:
        header.extend([f"{model}_DocID", f"{model}_Score", f"{model}_DocName"])
    rows = []
    for i in range(max_len):
        row = []
        for model in results_dict:
            if i < len(results_dict[model]):
                doc_id, score = results_dict[model][i]
                row.extend([doc_id, score, docs[doc_id].name])
            else:
                row.extend(["", "", ""])
        rows.append(row)
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Results saved to {filename}")

def main():
    ap = argparse.ArgumentParser(description="Classic IR Models: Boolean, VSM (TF–IDF), BM25")
    ap.add_argument("--corpus", type=str, default=None, help="Path to folder with .txt/.md files")
    ap.add_argument("--query", type=str, required=False, default="information retrieval models")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    if args.corpus:
        docs = load_corpus_from_folder(args.corpus)
    else:
        docs = demo_corpus()

    results_dict = {}
    bm = BooleanModel(docs)
    results_dict["Boolean"] = bm.query(args.query)
    vsm = VectorSpaceModel(docs)
    results_dict["VSM"] = vsm.query(args.query, top_k=args.topk)
    pb = BM25(docs)
    results_dict["BM25"] = pb.query(args.query, top_k=args.topk)

    for model, res in results_dict.items():
        print_results(model, docs, res, args.topk)

    save_results_csv("results.csv", docs, results_dict)

if __name__ == "__main__":
    main()
