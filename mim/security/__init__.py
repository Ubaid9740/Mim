"""Zero-Trust Security Masking Firewall for Mim.

Provides real-time process-level blindfolding, multi-stage text sanitization
(regex, AST code parsing, Shannon entropy analysis), and credential masking
to prevent secrets from ever reaching persistent storage substrates.
"""

from mim.security.process_guard import ProcessGuard
from mim.security.sanitizer import ZeroTrustSanitizer

# Backward-compatibility alias
TextSanitizer = ZeroTrustSanitizer

__all__ = ["ProcessGuard", "ZeroTrustSanitizer", "TextSanitizer"]

