import ast
import math
import re
from collections import Counter
from typing import List, Set, Tuple


class ZeroTrustSanitizer:
    """
    3-Stage Ingestion Firewall for Mim:
    - Stage 1: Regex pattern matching for structured API keys, JWTs, DSNs, and .env secrets.
    - Stage 2: Abstract Syntax Tree (AST) & lexical assignment masking for IDE code blocks.
    - Stage 3: Shannon Entropy scanning for unformatted cryptographic secrets.
    """

    REDACTION_TOKEN = "[REDACTED_SECRET]"

    IDE_PROCESSES: Set[str] = {
        "code.exe",
        "cursor.exe",
        "pycharm64.exe",
        "idea64.exe",
        "devenv.exe",
        "windsurf.exe",
        "antigravity.exe",
        "sublime_text.exe",
        "nvim.exe",
        "vim.exe",
    }

    SENSITIVE_IDENTIFIERS: Set[str] = {
        "password", "passwd", "pwd", "secret", "api_key", "apikey",
        "access_key", "secret_key", "private_key", "token", "auth_token",
        "refresh_token", "client_secret", "jwt_secret", "db_pass",
        "database_url", "redis_url", "dsn", "connection_string",
        "stripe_key", "openai_api_key", "gemini_api_key", "anthropic_api_key",
        "supabase_key", "neon_api_key", "encryption_key", "master_key",
    }

    def __init__(self, entropy_threshold: float = 4.35, min_token_len: int = 24):
        self.entropy_threshold = entropy_threshold
        self.min_token_len = min_token_len
        self._compiled_regexes: List[Tuple[re.Pattern, str]] = self._build_regex_rules()

    def _build_regex_rules(self) -> List[Tuple[re.Pattern, str]]:
        """Compiles high-precision regular expressions for standard credential formats."""
        r = self.REDACTION_TOKEN
        return [
            # PEM Private Keys
            (
                re.compile(
                    r"-----BEGIN [A-Z0-9\s]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9\s]*PRIVATE KEY-----"
                ),
                r,
            ),
            # Database Connection URLs (postgres://user:pass@host, mongodb+srv://...)
            (
                re.compile(
                    r"\b((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|rediss|amqp)://[^:\s/]+:)([^@\s]+)(@[^\s\"']+)",
                    re.IGNORECASE,
                ),
                rf"\1{r}\3",
            ),
            # Authorization Bearer / Basic Headers
            (
                re.compile(
                    r"(\b(?:Authorization|Bearer|Basic)\s*[:=]?\s*(?:Bearer\s+|Basic\s+)?)([A-Za-z0-9_\-\.~+/]+=*)",
                    re.IGNORECASE,
                ),
                rf"\1{r}",
            ),
            # JSON Web Tokens (JWT: eyJ... . eyJ... . sig)
            (
                re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
                r,
            ),
            # Google Cloud / Gemini API Keys (AIza...)
            (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), r),
            # AWS Access Key IDs
            (re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b"), r),
            # GitHub Personal Access & Fine-Grained Tokens
            (
                re.compile(
                    r"\b(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|gho_[A-Za-z0-9]{30,}|ghs_[A-Za-z0-9]{30,})\b"
                ),
                r,
            ),
            # OpenAI / Anthropic / Slack / Stripe Keys
            (re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"), r),
            (re.compile(r"\bsk-(?:proj-|live-|test-)?[A-Za-z0-9_-]{20,}\b"), r),
            (re.compile(r"\b(?:rk_live_|pk_live_|rk_test_)[0-9a-zA-Z]{20,}\b"), r),
            (re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,48}\b"), r),
            # .env / Shell / Config Key-Value Assignments
            (
                re.compile(
                    r"(?i)\b([A-Z0-9_]*(?:PASSWORD|PASSWD|SECRET|API_KEY|APIKEY|ACCESS_KEY|PRIVATE_KEY|TOKEN|AUTH|DSN|DATABASE_URL)[A-Z0-9_]*)"
                    r"(\s*[:=]\s*)"
                    r"(\"[^\"]+\"|'[^']+'|[^\s#\"']+)"
                ),
                rf'\1\2"{r}"',
            ),
        ]

    def _is_sensitive_identifier(self, name: str) -> bool:
        """Checks if a variable name or dictionary key denotes a secret."""
        clean = name.strip().lower()
        if clean in self.SENSITIVE_IDENTIFIERS:
            return True
        return any(sub in clean for sub in self.SENSITIVE_IDENTIFIERS)

    def _stage1_regex_pass(self, text: str) -> str:
        """Applies compiled regex signatures across the text."""
        sanitized = text
        for pattern, replacement in self._compiled_regexes:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized

    def _stage2_ast_code_pass(self, text: str) -> str:
        """
        Parses Python code via `ast` to mask literal values assigned to sensitive identifiers
        while keeping the surrounding syntax intact. Falls back to lexical assignment scanning
        for partial snippets or non-Python languages (JS/TS/JSON).
        """
        spans_to_redact: List[Tuple[int, int]] = []
        lines = text.splitlines(keepends=True)

        # Calculate line start character offsets for exact AST node slicing
        line_offsets = [0]
        for line in lines:
            line_offsets.append(line_offsets[-1] + len(line))

        def record_node_span(val_node: ast.AST) -> None:
            if isinstance(val_node, ast.Constant) and isinstance(val_node.value, str):
                if val_node.value == self.REDACTION_TOKEN or len(val_node.value.strip()) == 0:
                    return
                if (
                    hasattr(val_node, "lineno")
                    and hasattr(val_node, "end_lineno")
                    and val_node.lineno is not None
                    and val_node.end_lineno is not None
                ):
                    start_idx = line_offsets[val_node.lineno - 1] + (val_node.col_offset or 0)
                    end_idx = line_offsets[val_node.end_lineno - 1] + (val_node.end_col_offset or 0)
                    spans_to_redact.append((start_idx, end_idx))

        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                # Standard Assignments: db_password = "..."
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and self._is_sensitive_identifier(target.id):
                            record_node_span(node.value)
                        elif isinstance(target, ast.Attribute) and self._is_sensitive_identifier(target.attr):
                            record_node_span(node.value)
                        elif isinstance(target, ast.Subscript):
                            if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                                if self._is_sensitive_identifier(target.slice.value):
                                    record_node_span(node.value)

                # Type-Annotated Assignments: api_key: str = "..."
                elif isinstance(node, ast.AnnAssign):
                    if isinstance(node.target, ast.Name) and self._is_sensitive_identifier(node.target.id):
                        if node.value:
                            record_node_span(node.value)

                # Keyword Arguments in Function Calls: connect(password="...")
                elif isinstance(node, ast.keyword):
                    if node.arg and self._is_sensitive_identifier(node.arg):
                        record_node_span(node.value)

                # Dictionary Literals: {"client_secret": "..."}
                elif isinstance(node, ast.Dict):
                    for k, v in zip(node.keys, node.values):
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            if self._is_sensitive_identifier(k.value):
                                record_node_span(v)
        except SyntaxError:
            # Expected when scraping partial viewport code or JS/TS/Rust/JSON
            pass

        # Apply AST span replacements in reverse order so character offsets stay valid
        if spans_to_redact:
            spans_to_redact.sort(key=lambda x: x[0], reverse=True)
            chars = list(text)
            for start, end in spans_to_redact:
                chars[start:end] = list(f'"{self.REDACTION_TOKEN}"')
            text = "".join(chars)

        # Lexical fallback for JS/TS/JSON or unparseable partial Python blocks
        lexical_pattern = re.compile(
            r"(?i)(['\"]?[a-zA-Z0-9_]*(?:password|passwd|pwd|secret|api_key|apikey|token|private_key|db_pass|dsn|database_url)[a-zA-Z0-9_]*['\"]?)"
            r"(\s*(?::|=)\s*)"
            r"(\"[^\"\n]*\"|'[^'\n]*')"
        )
        text = lexical_pattern.sub(rf'\1\2"{self.REDACTION_TOKEN}"', text)
        return text

    @staticmethod
    def calculate_shannon_entropy(token: str) -> float:
        """Computes the Shannon entropy (bits per character) of a string."""
        if not token:
            return 0.0
        length = len(token)
        counts = Counter(token)
        return -sum((count / length) * math.log2(count / length) for count in counts.values())

    def _looks_like_high_entropy_secret(self, token: str) -> bool:
        """Heuristically distinguishes high-entropy cryptographic strings from file paths or URLs."""
        clean = token.strip("\"'`,;()[]{}")
        if len(clean) < self.min_token_len:
            return False
        if self.REDACTION_TOKEN in clean:
            return False

        # Skip URLs, local filesystem paths, and standard namespace imports
        if clean.startswith(("http://", "https://", "file://", "C:\\", "D:\\", "/", "./", "../")):
            return False
        if "\\" in clean or clean.count("/") > 1 or clean.count(".") > 2:
            return False

        # Must have character class diversity (e.g., upper + lower + digits, or hex)
        has_lower = any(c.islower() for c in clean)
        has_upper = any(c.isupper() for c in clean)
        has_digit = any(c.isdigit() for c in clean)
        class_count = sum([has_lower, has_upper, has_digit])
        if class_count < 2:
            return False

        # Reject standard CamelCase/snake_case variable names that lack digits
        if not has_digit and ("_" in clean or clean.isalpha()):
            return False

        entropy = self.calculate_shannon_entropy(clean)

        # Hexadecimal secrets have max theoretical entropy of 4.0 bits/char (log2(16)).
        # Detect pure hex strings (length >= 32) using an adjusted threshold of 3.5 bits/char.
        is_hex = bool(re.fullmatch(r"[0-9a-fA-F]+", clean))
        if is_hex and len(clean) >= 32:
            return entropy >= 3.5

        return entropy >= self.entropy_threshold

    def _stage3_entropy_pass(self, text: str) -> str:
        """Scans contiguous non-whitespace tokens and masks unformatted high-entropy secrets."""
        def replace_token(match: re.Match) -> str:
            candidate = match.group(0)
            if self._looks_like_high_entropy_secret(candidate):
                return self.REDACTION_TOKEN
            return candidate

        # Match contiguous alphanumeric/symbol strings of length >= min_token_len
        token_regex = re.compile(rf"[A-Za-z0-9_\-\+/=]{{{self.min_token_len},}}")
        return token_regex.sub(replace_token, text)

    def sanitize(self, text: str, process_name: str = "") -> str:
        """
        Executes the full Zero-Trust sanitization pipeline on scraped UIA text.
        """
        if not text or not text.strip():
            return text

        # Stage 1: Fast Regex Pass
        result = self._stage1_regex_pass(text)

        # Stage 2: AST & Lexical Code Pass (Always run for IDEs or code-like blocks)
        proc_lower = (process_name or "").strip().lower()
        if proc_lower in self.IDE_PROCESSES or any(sym in result for sym in ("=", ":", "{")):
            result = self._stage2_ast_code_pass(result)

        # Stage 3: High-Entropy Secret Catch-All
        result = self._stage3_entropy_pass(result)

        return result