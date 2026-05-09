"""A multi-island experience buffer that implements the evolutionary algorithm."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import dataclasses
import time
from typing import Any, Tuple, Mapping

from absl import logging
import numpy as np
import scipy

from codes.update import code_manipulation
from codes import config as config_lib
from codes.evaluate import profile as exp_profile

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import ast

# -----------------------------
# AST-based canonical tokenizer
# -----------------------------

_SAFE_FUNCS = {
    # common math / numpy names that may appear as calls
    "sin", "cos", "tan", "exp", "log", "sqrt", "abs",
    "maximum", "minimum", "where", "clip", "tanh", "sinh", "cosh",
    "pow",
}

def _strip_comments(text: str) -> str:
    # remove single-line comments
    return re.sub(r"#.*", "", text)

def _extract_return_expr(code: str) -> ast.AST | None:
    """
    Parse code (usually a 'def equation(...): ... return ...') and extract the returned expression AST.
    If parsing fails, return None.
    """
    code = _strip_comments(code).strip()
    if not code:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    # find first return expression (prefer inside first FunctionDef)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for n in ast.walk(node):
                if isinstance(n, ast.Return) and n.value is not None:
                    return n.value
            break

    # fallback: any return in module
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            return node.value
    return None

def _extract_func_args(code: str) -> list[str]:
    code = _strip_comments(code).strip()
    if not code:
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = []
            for a in node.args.args:
                if a.arg not in {"self"}:
                    args.append(a.arg)
            return args
    return []

def _is_number_node(n: ast.AST) -> bool:
    return isinstance(n, ast.Constant) and isinstance(n.value, (int, float))

def _const_int_value(n: ast.AST) -> int | None:
    if isinstance(n, ast.Constant) and isinstance(n.value, int):
        return int(n.value)
    return None

def _dump_key(n: ast.AST) -> str:
    # stable structural key for sorting commutative children
    return ast.dump(n, annotate_fields=False, include_attributes=False)

def _flatten_binop(node: ast.AST, op_type: type) -> list[ast.AST]:
    """Flatten a chain of the same commutative BinOp (Add or Mult)."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, op_type):
        return _flatten_binop(node.left, op_type) + _flatten_binop(node.right, op_type)
    return [node]

def _make_binop_chain(items: list[ast.AST], op: ast.operator) -> ast.AST:
    """Rebuild BinOp chain from list (left-associated)."""
    assert len(items) >= 1
    expr = items[0]
    for it in items[1:]:
        expr = ast.BinOp(left=expr, op=op, right=it)
    return expr

def _canonicalize(node: ast.AST) -> ast.AST:
    """
    Canonicalize expression AST to reduce 'syntactic skins':
    - flatten & sort commutative operations (Add, Mult)
    - convert repeated multiplication into Pow (x*x -> x**2; x*x*x -> x**3)
    - normalize pow(x,2) call into Pow(x,2)
    """
    if node is None:
        return node

    # Recursively canonicalize children first
    if isinstance(node, ast.BinOp):
        node.left = _canonicalize(node.left)
        node.right = _canonicalize(node.right)

        # (1) Canonicalize addition: flatten and sort
        if isinstance(node.op, ast.Add):
            items = [_canonicalize(x) for x in _flatten_binop(node, ast.Add)]
            items.sort(key=_dump_key)
            return _make_binop_chain(items, ast.Add())

        # (2) Canonicalize multiplication: flatten and sort, then compress repeats to Pow
        if isinstance(node.op, ast.Mult):
            items = [_canonicalize(x) for x in _flatten_binop(node, ast.Mult)]
            # sort first so repeats are adjacent
            items.sort(key=_dump_key)

            # compress identical factors: a*a*a -> a**3
            compressed: list[ast.AST] = []
            i = 0
            while i < len(items):
                base = items[i]
                j = i + 1
                while j < len(items) and _dump_key(items[j]) == _dump_key(base):
                    j += 1
                count = j - i
                if count >= 2:
                    compressed.append(ast.BinOp(left=base, op=ast.Pow(), right=ast.Constant(count)))
                else:
                    compressed.append(base)
                i = j

            compressed.sort(key=_dump_key)
            return _make_binop_chain(compressed, ast.Mult())

        return node

    if isinstance(node, ast.UnaryOp):
        node.operand = _canonicalize(node.operand)
        return node

    if isinstance(node, ast.Call):
        # normalize pow(x,2) -> x**2
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # e.g., np.exp, math.exp
            func_name = node.func.attr

        args = [_canonicalize(a) for a in node.args]
        if func_name == "pow" and len(args) == 2:
            return ast.BinOp(left=args[0], op=ast.Pow(), right=args[1])

        # canonicalize other call arguments; keep call itself
        node.args = args
        return node

    if isinstance(node, ast.Attribute):
        # canonicalize base
        node.value = _canonicalize(node.value)
        return node

    # Name / Constant / others: return as-is
    return node

def _collect_name_counts(expr: ast.AST) -> dict[str, int]:
    counts: dict[str, int] = {}
    for n in ast.walk(expr):
        if isinstance(n, ast.Name):
            counts[n.id] = counts.get(n.id, 0) + 1
    return counts

def _build_role_map(expr: ast.AST, func_args: list[str]) -> dict[str, str]:
    """
    Role mapping ONLY over function arguments (true inputs).
    - t/time -> VAR_TIME (excluded from VAR_SELF competition)
    - most frequent remaining arg -> VAR_SELF
    - others -> VAR_OTHER
    """
    arg_set = set(func_args)

    counts = {a: 0 for a in func_args}
    for n in ast.walk(expr):
        if isinstance(n, ast.Name) and n.id in arg_set:
            counts[n.id] += 1

    role: dict[str, str] = {}

    for tname in ("t", "time"):
        if tname in counts:
            role[tname] = "VAR_TIME"
            counts.pop(tname, None)

    if not counts:
        return role

    main = max(counts.items(), key=lambda kv: kv[1])[0]
    role[main] = "VAR_SELF"
    for k in counts:
        if k != main:
            role[k] = "VAR_OTHER"
    return role

def _emit_tokens(expr: ast.AST, role_map: dict[str, str], out: list[str]) -> None:
    """
    Preorder traversal to emit stable tokens.
    """
    if expr is None:
        return

    if isinstance(expr, ast.BinOp):
        if isinstance(expr.op, ast.Add): out.append("ADD")
        elif isinstance(expr.op, ast.Sub): out.append("SUB")
        elif isinstance(expr.op, ast.Mult): out.append("MUL")
        elif isinstance(expr.op, ast.Div): out.append("DIV")
        elif isinstance(expr.op, ast.Pow): out.append("POW")
        else: out.append("BINOP")

        _emit_tokens(expr.left, role_map, out)
        _emit_tokens(expr.right, role_map, out)
        return

    if isinstance(expr, ast.UnaryOp):
        if isinstance(expr.op, ast.USub): out.append("NEG")
        else: out.append("UNARY")
        _emit_tokens(expr.operand, role_map, out)
        return

    if isinstance(expr, ast.Call):
        func_name = None
        if isinstance(expr.func, ast.Name):
            func_name = expr.func.id
        elif isinstance(expr.func, ast.Attribute):
            func_name = expr.func.attr
        if func_name is None:
            func_name = "CALL"
        out.append(f"FUNC_{func_name}")
        for a in expr.args:
            _emit_tokens(a, role_map, out)
        return

    if isinstance(expr, ast.Name):
        out.append(role_map.get(expr.id, "VAR_OTHER"))
        return

    if isinstance(expr, ast.Constant):
        if isinstance(expr.value, (int, float)):
            # keep small integer powers distinct if you want (helps x**2 vs x**3)
            if isinstance(expr.value, int) and -5 <= expr.value <= 5:
                out.append(f"INT_{expr.value}")
            else:
                out.append("NUM")
        else:
            out.append("CONST")
        return

    if isinstance(expr, ast.Attribute):
        # e.g., np.exp -> treat as FUNC_exp handled in Call; attribute alone rare
        out.append(f"ATTR_{expr.attr}")
        _emit_tokens(expr.value, role_map, out)
        return

    # fallback
    out.append(expr.__class__.__name__)

def _math_tokenizer(text: str) -> list[str]:
    """
    Canonical tokenizer for TF-IDF:
    - AST parse return expression
    - canonicalize commutative forms and x*x -> x**2
    - role-map variables into VAR_SELF/VAR_OTHER/VAR_TIME
    - emit preorder tokens
    """
    expr = _extract_return_expr(text)
    if expr is None:
        # fallback to old regex tokens, but still remove comments
        text2 = _strip_comments(text)
        return re.findall(r"(?u)\b[a-zA-Z_]\w*\b|[+\-*/^()]", text2)

    expr = _canonicalize(expr)
    func_args = _extract_func_args(text)
    role_map = _build_role_map(expr, func_args)

    tokens: list[str] = []
    _emit_tokens(expr, role_map, tokens)

    return tokens

class SemanticFilter:
    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(
            tokenizer=_math_tokenizer,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
        )
        self.corpus = []

    def check_similarity(self, new_code):
        """Return (is_duplicate, best_index, similarity)."""
        if not self.corpus:
            return False, -1, 0.0

        try:
            temp_corpus = self.corpus + [new_code]
            tfidf_matrix = self.vectorizer.fit_transform(temp_corpus)
            
            new_vec = tfidf_matrix[-1]
            history_vecs = tfidf_matrix[:-1]
            
            sims = cosine_similarity(new_vec, history_vecs).flatten()
            
            if len(sims) == 0:
                return False, -1, 0.0
                
            best_idx = np.argmax(sims)
            best_sim = sims[best_idx]
            
            if best_sim > self.threshold:
                return True, best_idx, best_sim
            else:
                return False, best_idx, best_sim
                
        except ValueError:
            return False, -1, 0.0

    def add_to_corpus(self, code):
        self.corpus.append(code)
        
    def update_corpus_at_index(self, index, code):
        if 0 <= index < len(self.corpus):
            self.corpus[index] = code

Signature = Tuple[float, ...]
ScoresPerTest = Mapping[Any, float]


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite `logits`."""
    if not np.all(np.isfinite(logits)):
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)

    result = scipy.special.softmax(logits / temperature, axis=-1)
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1:])
    return result


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
    test_scores = [scores_per_test[k] for k in scores_per_test.keys()]
    return sum(test_scores) / len(test_scores)


def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
    """Represents test scores as a canonical signature."""
    return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


@dataclasses.dataclass(frozen=True)
class Prompt:
    """ A prompt produced by the Experience Buffer, to be sent to Samplers.

    Args:
      code: The prompt, ending with the header of the function to be completed.
      version_generated: The function to be completed is `_v{version_generated}`.
      island_id: Identifier of the island that produced the samples
                included in the prompt. Used to direct the newly generated sample
                into the same island.
    """
    code: str
    version_generated: int
    island_id: int

class ExperienceBuffer:
    """A collection of programs, organized as islands."""

    def __init__(
            self,
            config: config_lib.ExperienceBufferConfig,
            template: code_manipulation.Program,
            function_to_evolve: str,
    ) -> None:
        self._config: config_lib.ExperienceBufferConfig = config
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve

        # Initialize empty islands.
        self._islands: list[Island] = []
        for _ in range(config.num_islands):
            self._islands.append(
                Island(
                    template,
                    function_to_evolve,
                    config.functions_per_prompt,
                    config.cluster_sampling_temperature_init,
                    config.cluster_sampling_temperature_period,
                    use_tfidf_for_clustering=config.use_tfidf_for_clustering,
                ))
        self._best_score_per_island: list[float] = (
                [-float('inf')] * config.num_islands)
        self._best_program_per_island: list[code_manipulation.Function | None] = (
                [None] * config.num_islands)
        self._best_scores_per_test_per_island: list[ScoresPerTest | None] = (
                [None] * config.num_islands)

        self._last_reset_time: float = time.time()


    def get_prompt(self) -> Prompt:
        """Returns a prompt containing samples from one chosen island."""
        island_id = np.random.randint(len(self._islands))
        code, version_generated = self._islands[island_id].get_prompt()
        return Prompt(code, version_generated, island_id)


    def _register_program_in_island(
            self,
            program: code_manipulation.Function,
            island_id: int,
            scores_per_test: ScoresPerTest,
            **kwargs 
    ) -> None:
        """Registers `program` in the specified island."""
        try:
            setattr(program, "island_id", int(island_id))
        except Exception:
            pass
        self._islands[island_id].register_program(program, scores_per_test)
        score = _reduce_score(scores_per_test)
        if score > self._best_score_per_island[island_id]:
            self._best_program_per_island[island_id] = program
            self._best_scores_per_test_per_island[island_id] = scores_per_test
            self._best_score_per_island[island_id] = score
            logging.info('Best score of island %d increased to %s', island_id, score)
        profiler: exp_profile.Profiler = kwargs.get('profiler', None)
        if profiler:
            sample_order = kwargs.get("sample_order", None)  
            global_sample_nums = kwargs.get('global_sample_nums', None)
            sample_time = kwargs.get('sample_time', None)
            evaluate_time = kwargs.get('evaluate_time', None)

            program.score = score

            program.global_sample_nums = sample_order if sample_order is not None else global_sample_nums

            if "refine" in kwargs:
                try:
                    setattr(program, "refine", kwargs.get("refine"))
                except Exception:
                    pass
            if "stage" in kwargs:
                try:
                    setattr(program, "stage", kwargs.get("stage"))
                except Exception:
                    pass

            program.sample_time = sample_time
            program.evaluate_time = evaluate_time
            profiler.register_function(program)


    def register_program(
            self,
            program: code_manipulation.Function,
            island_id: int | None,
            scores_per_test: ScoresPerTest,
            **kwargs 
    ) -> None:
        """Registers new `program` skeleton hypotheses in the experience buffer."""
        if island_id is None:
            for island_id in range(len(self._islands)):
                self._register_program_in_island(program, island_id, scores_per_test, **kwargs)
        else:
            self._register_program_in_island(program, island_id, scores_per_test, **kwargs)

        # Check island reset
        if time.time() - self._last_reset_time > self._config.reset_period:
            self._last_reset_time = time.time()
            self.reset_islands()


    def reset_islands(self) -> None:
        """Resets the weaker half of islands."""
        indices_sorted_by_score: np.ndarray = np.argsort(
            self._best_score_per_island +
            np.random.randn(len(self._best_score_per_island)) * 1e-6)
        num_islands_to_reset = self._config.num_islands // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for island_id in reset_islands_ids:
            self._islands[island_id] = Island(
                self._template,
                self._function_to_evolve,
                self._config.functions_per_prompt,
                self._config.cluster_sampling_temperature_init,
                self._config.cluster_sampling_temperature_period,
                use_tfidf_for_clustering=self._config.use_tfidf_for_clustering,
            )
            self._best_score_per_island[island_id] = -float('inf')
            founder_island_id = np.random.choice(keep_islands_ids)
            founder = self._best_program_per_island[founder_island_id]
            founder_scores = self._best_scores_per_test_per_island[founder_island_id]
            self._register_program_in_island(founder, island_id, founder_scores)

class Island:
    """A sub-population of the program skeleton experience buffer."""
    def __init__(
            self,
            template: code_manipulation.Program,
            function_to_evolve: str,
            functions_per_prompt: int,
            cluster_sampling_temperature_init: float,
            cluster_sampling_temperature_period: int,
            use_tfidf_for_clustering: bool = True,
    ) -> None:
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve
        self._functions_per_prompt: int = functions_per_prompt
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = cluster_sampling_temperature_period
        self._use_tfidf_for_clustering = use_tfidf_for_clustering

        self._clusters: list[Cluster] = []
        self._num_programs: int = 0
        self.semantic_filter: SemanticFilter | None = (
            SemanticFilter(threshold=0.90) if use_tfidf_for_clustering else None
        )

    def register_program(
                self,
                program: code_manipulation.Function,
                scores_per_test: ScoresPerTest,
        ) -> None:
            """Stores a program on this island. Uses TF-IDF similarity or score signature depending on config."""
            program_score = _reduce_score(scores_per_test)
            program.score = program_score

            if self._use_tfidf_for_clustering and self.semantic_filter is not None:
                # TF-IDF mode: cluster by code-structure similarity.
                code_content = program.body
                is_duplicate, similar_idx, _ = self.semantic_filter.check_similarity(code_content)
                if is_duplicate:
                    target_cluster = self._clusters[similar_idx]
                    target_cluster.register_program(program)
                    try:
                        setattr(program, "cluster_id", int(similar_idx))
                    except Exception:
                        pass
                    if program.score >= target_cluster.score:
                        self.semantic_filter.update_corpus_at_index(similar_idx, code_content)
                else:
                    new_cluster = Cluster(program_score, program, max_size=20)
                    self._clusters.append(new_cluster)
                    self.semantic_filter.add_to_corpus(code_content)
                    try:
                        setattr(program, "cluster_id", int(len(self._clusters) - 1))
                    except Exception:
                        pass
            else:
                # Score-signature mode (original LLMSR): same signature -> same cluster.
                sig = _get_signature(scores_per_test)
                target_idx = None
                for i, c in enumerate(self._clusters):
                    if c._signature == sig:
                        target_idx = i
                        break
                if target_idx is not None:
                    self._clusters[target_idx].register_program(program)
                    try:
                        setattr(program, "cluster_id", int(target_idx))
                    except Exception:
                        pass
                else:
                    new_cluster = Cluster(program_score, program, max_size=20, signature=sig)
                    self._clusters.append(new_cluster)
                    try:
                        setattr(program, "cluster_id", int(len(self._clusters) - 1))
                    except Exception:
                        pass

            self._num_programs += 1

    def get_prompt(self) -> tuple[str, int]:
        """Constructs a prompt containing equation program skeletons from this island."""
        # Check if clusters are empty
        if len(self._clusters) == 0:
            # Return empty prompt if no clusters exist (should not happen in normal flow)
            # This can occur if initial program evaluation failed
            raise RuntimeError(
                "Cannot generate prompt: no clusters in island. "
                "This usually means the initial program evaluation failed or returned invalid scores. "
                "Please check the evaluate function in your specification file."
            )
        
        # Cluster weight uses each cluster's best score (semantic: elite within TF-IDF group).
        cluster_scores = np.array([cluster.score for cluster in self._clusters])
        # Temperature decays within a period (more greedy), then resets for exploration.
        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * (
                1 - (self._num_programs % period) / period)
        probabilities = _softmax(cluster_scores, temperature)
        # Sample clusters with replacement; cap count by available clusters.
        functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)
        cluster_indices = np.random.choice(
            len(self._clusters), size=functions_per_prompt, p=probabilities)
        implementations = []
        scores = []
        for idx in cluster_indices:
            cluster = self._clusters[idx]
            if self._use_tfidf_for_clustering:
                # Semantic cluster: always expose the highest-scoring program as the elite exemplar.
                implementations.append(cluster.best_program)
            else:
                # Score-signature clusters share the same signature; keep length-biased diversity.
                implementations.append(cluster.sample_program())
            scores.append(cluster.score)

        indices = np.argsort(scores)
        sorted_implementations = [implementations[i] for i in indices]
        version_generated = len(sorted_implementations) + 1
        return self._generate_prompt(sorted_implementations), version_generated

    def _generate_prompt(
            self,
            implementations: Sequence[code_manipulation.Function]) -> str:
        """ Create a prompt containing a sequence of function `implementations`."""
        implementations = copy.deepcopy(implementations)

        # Format the names and docstrings of functions to be included in the prompt.
        versioned_functions: list[code_manipulation.Function] = []
        for i, implementation in enumerate(implementations):
            new_function_name = f'{self._function_to_evolve}_v{i}'
            implementation.name = new_function_name
            # Update the docstring for all subsequent functions after `_v0`.
            if i >= 1:
                implementation.docstring = (
                    f'Improved version of `{self._function_to_evolve}_v{i - 1}`.')
            # If recursive, rename self-calls to the versioned name.
            implementation = code_manipulation.rename_function_calls(
                str(implementation), self._function_to_evolve, new_function_name)
            versioned_functions.append(
                code_manipulation.text_to_function(implementation))

        # Header: empty body for the LLM to complete.
        next_version = len(implementations)
        new_function_name = f'{self._function_to_evolve}_v{next_version}'
        header = dataclasses.replace(
            implementations[-1],
            name=new_function_name,
            body='',
            docstring=('Improved version of '
                       f'`{self._function_to_evolve}_v{next_version - 1}`.'),
        )
        versioned_functions.append(header)

        # Replace functions in the template with the list constructed here.
        prompt = dataclasses.replace(self._template, functions=versioned_functions)
        
        return str(prompt)

class Cluster:
    """ A cluster of programs with similar semantic structure (TF-IDF) or same score signature (score-based). """

    def __init__(
        self,
        score: float,
        implementation: code_manipulation.Function,
        max_size: int = 20,
        signature: Signature | None = None,
    ):
        self._max_size = max_size
        self._programs: list[code_manipulation.Function] = [implementation]
        self._lengths: list[int] = [len(str(implementation))]
        self._score = score
        self._signature: Signature | None = signature  # score-based clustering only

    @property
    def score(self) -> float:
        if not self._programs:
            return -float('inf')
        return max(p.score for p in self._programs if p.score is not None)
    
    @property
    def best_program(self) -> code_manipulation.Function:
        return max(self._programs, key=lambda p: p.score if p.score is not None else -float('inf'))

    def register_program(self, program: code_manipulation.Function) -> None:
        """Adds `program` to the cluster, maintaining Top-K."""
        
        self._programs.append(program)
        self._lengths.append(len(str(program)))
        
        combined = list(zip(self._programs, self._lengths))
        combined.sort(key=lambda x: x[0].score if x[0].score is not None else -float('inf'), reverse=True)
        
        if len(combined) > self._max_size:
            combined = combined[:self._max_size]
            
        self._programs, self._lengths = zip(*combined)
        self._programs = list(self._programs)
        self._lengths = list(self._lengths)
        
        self._score = self._programs[0].score

    def sample_program(self) -> code_manipulation.Function:
        """Samples a program, giving higher probability to shorther programs."""
        normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
                max(self._lengths) + 1e-6)
        # Prefer shorter programs (simpler equations).
        probabilities = _softmax(-normalized_lengths, temperature=1.0)
        return np.random.choice(self._programs, p=probabilities)
