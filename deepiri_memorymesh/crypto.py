"""Encryption primitives for Memory Mesh field-level encryption (T33).

This module is the cryptographic foundation only: envelope format, key
loading/generation, HKDF-derived subkeys, and keyed HMAC helpers. Enable/
rotate orchestration against the SQLite store lives in :mod:`encryption`.

Design notes:

- ``cryptography`` is an *optional* dependency (``pip install
  memorymesh[security]``). Importing this module never fails even when it is
  missing; any function that actually needs AEAD or HKDF calls
  :func:`require_cryptography` first so the failure is immediate and clear.
- Ciphertext is stored as a small versioned JSON envelope (string) so it can
  live in existing ``TEXT`` columns without a schema shape change:
  ``{"v": 1, "alg": "AES-256-GCM", "kid": "<key_id>", "nonce": "<b64>", "ct": "<b64>"}``.
- The AEAD additional-authenticated-data (AAD) binds each ciphertext to
  ``"{db_identity}|{table}|{column}|{row_identity}"`` so a ciphertext copied
  to a different database, table, column, or row fails to decrypt instead of
  silently decrypting into the wrong place.
- Key material is never logged or printed anywhere in this module.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised via mocking in tests when unavailable
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives import hashes as _hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    CRYPTOGRAPHY_AVAILABLE = True
    _CRYPTOGRAPHY_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # noqa: BLE001 - deliberately broad: any import failure
    InvalidTag = None  # type: ignore[assignment]
    _hashes = None  # type: ignore[assignment]
    AESGCM = None  # type: ignore[assignment]
    HKDF = None  # type: ignore[assignment]
    CRYPTOGRAPHY_AVAILABLE = False
    _CRYPTOGRAPHY_IMPORT_ERROR = _exc


DEFAULT_KEY_ENV_VAR = "MEMORYMESH_ENCRYPTION_KEY"
ENVELOPE_VERSION = 1
ALGORITHM = "AES-256-GCM"
MIN_KEY_BYTES = 32
NONCE_BYTES = 12

_HKDF_INFO_ENC = b"memorymesh:content-enc:v1"
_HKDF_INFO_CONTENT_HMAC = b"memorymesh:content-hmac:v1"
_HKDF_INFO_TERM_HMAC = b"memorymesh:term-hmac:v1"

_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


class CryptoError(RuntimeError):
    """Base class for all Memory Mesh encryption failures."""


class CryptographyUnavailableError(CryptoError):
    """Raised when encryption is requested but ``cryptography`` isn't installed."""


class InvalidKeyMaterialError(CryptoError):
    """Raised for missing, malformed, or too-short key material."""


class KeyFileExistsError(CryptoError):
    """Raised by :func:`generate_key_file` when the target path already exists."""


class MalformedEnvelopeError(CryptoError):
    """Raised when a stored value is not a well-formed encryption envelope."""


class UnsupportedEnvelopeVersionError(CryptoError):
    """Raised when an envelope's ``v`` field is not one this build supports."""


class WrongKeyError(CryptoError):
    """Raised when an envelope's ``kid`` does not match the active key."""


class TamperError(CryptoError):
    """Raised when AEAD authentication fails against the *matching* key id.

    Since the key id matched but the authentication tag did not verify, the
    ciphertext, nonce, or associated data (table/column/row binding) has been
    altered or corrupted.
    """


def require_cryptography() -> None:
    """Raise :class:`CryptographyUnavailableError` with a clear fix if missing."""
    if CRYPTOGRAPHY_AVAILABLE:
        return
    detail = f" (import error: {_CRYPTOGRAPHY_IMPORT_ERROR})" if _CRYPTOGRAPHY_IMPORT_ERROR else ""
    raise CryptographyUnavailableError(
        "Encryption requires the optional 'cryptography' package, which is not "
        "installed. Install it with: pip install 'memorymesh[security]' "
        "(or: pip install 'cryptography>=42.0.0')." + detail
    )


def _decode_key_material(raw: str) -> bytes:
    """Decode base64url or hex key text into raw bytes (>= 32 bytes required)."""
    text = raw.strip()
    if not text:
        raise InvalidKeyMaterialError("Key material is empty.")

    if len(text) % 2 == 0 and _HEX_RE.match(text):
        try:
            data = bytes.fromhex(text)
        except ValueError as exc:
            raise InvalidKeyMaterialError(f"Key material looks like hex but failed to decode: {exc}") from exc
        if len(data) < MIN_KEY_BYTES:
            raise InvalidKeyMaterialError(
                f"Decoded hex key material is {len(data)} bytes; at least {MIN_KEY_BYTES} required."
            )
        return data

    padded = text + "=" * (-len(text) % 4)
    try:
        data = base64.urlsafe_b64decode(padded)
    except (binascii.Error, ValueError) as exc:
        raise InvalidKeyMaterialError(
            "Key material is neither valid hex nor valid base64url."
        ) from exc
    if len(data) < MIN_KEY_BYTES:
        raise InvalidKeyMaterialError(
            f"Decoded key material is {len(data)} bytes; at least {MIN_KEY_BYTES} required."
        )
    return data


def load_key_from_env(env_var: str = DEFAULT_KEY_ENV_VAR) -> bytes:
    """Load and decode key material from an environment variable."""
    raw = os.environ.get(env_var)
    if raw is None or not raw.strip():
        raise InvalidKeyMaterialError(
            f"Environment variable {env_var!r} is not set. Set it to a base64url- "
            f"or hex-encoded key of at least {MIN_KEY_BYTES} bytes, or pass a key file."
        )
    return _decode_key_material(raw)


def load_key_from_file(path: Path | str) -> bytes:
    """Load and decode key material from a key file (as written by ``generate_key_file``)."""
    file_path = Path(path)
    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InvalidKeyMaterialError(f"Could not read key file {file_path}: {exc}") from exc
    return _decode_key_material(raw)


def load_master_key(
    *,
    env_var: str = DEFAULT_KEY_ENV_VAR,
    key_file: Path | str | None = None,
) -> bytes:
    """Resolve the master key: explicit *key_file* wins, else *env_var*.

    Raises :class:`InvalidKeyMaterialError` with a clear message when neither
    source yields valid key material.
    """
    if key_file is not None:
        return load_key_from_file(key_file)
    return load_key_from_env(env_var)


def generate_key_file(path: Path | str, *, overwrite: bool = False) -> Path:
    """Write 32 fresh random bytes (base64-encoded) to *path* with mode 0o600.

    Fails if *path* already exists unless *overwrite* is ``True``. The key
    bytes are never returned, logged, or printed by this function.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = base64.b64encode(secrets.token_bytes(MIN_KEY_BYTES)).decode("ascii")

    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if overwrite else os.O_EXCL)
    try:
        fd = os.open(file_path, flags, 0o600)
    except FileExistsError as exc:
        raise KeyFileExistsError(
            f"Key file already exists: {file_path}. Pass overwrite=True to replace it."
        ) from exc
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
    finally:
        pass
    os.chmod(file_path, 0o600)
    return file_path


def _hkdf(key_material: bytes, info: bytes, length: int = 32) -> bytes:
    require_cryptography()
    derived = HKDF(algorithm=_hashes.SHA256(), length=length, salt=None, info=info).derive(key_material)
    return derived


def key_id_for(master_key: bytes) -> str:
    """Non-secret key identifier: first 16 hex chars of sha256(key)."""
    return hashlib.sha256(master_key).hexdigest()[:16]


@dataclass(slots=True)
class EncryptionContext:
    """Holds the master key and its HKDF-derived subkeys for one database.

    Constructing this requires ``cryptography`` (subkey derivation uses
    HKDF-SHA256) and raises :class:`CryptographyUnavailableError` clearly
    when it is missing.
    """

    master_key: bytes
    db_identity: str
    key_id: str = field(init=False)
    _enc_key: bytes = field(init=False, repr=False)
    _content_hmac_key: bytes = field(init=False, repr=False)
    _term_hmac_key: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.master_key) < MIN_KEY_BYTES:
            raise InvalidKeyMaterialError(
                f"Master key is {len(self.master_key)} bytes; at least {MIN_KEY_BYTES} required."
            )
        if not self.db_identity:
            raise InvalidKeyMaterialError("db_identity must be a non-empty string.")
        self.key_id = key_id_for(self.master_key)
        self._enc_key = _hkdf(self.master_key, _HKDF_INFO_ENC)
        self._content_hmac_key = _hkdf(self.master_key, _HKDF_INFO_CONTENT_HMAC)
        self._term_hmac_key = _hkdf(self.master_key, _HKDF_INFO_TERM_HMAC)

    @property
    def content_hmac_key(self) -> bytes:
        return self._content_hmac_key

    @property
    def term_hmac_key(self) -> bytes:
        return self._term_hmac_key


def _aad(ctx: EncryptionContext, *, table: str, column: str, row_identity: str) -> bytes:
    return f"{ctx.db_identity}|{table}|{column}|{row_identity}".encode("utf-8")


def _encrypt_raw(ctx: EncryptionContext, plaintext: str, *, aad: bytes) -> str:
    require_cryptography()
    nonce = os.urandom(NONCE_BYTES)
    aesgcm = AESGCM(ctx._enc_key)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), aad)
    envelope = {
        "v": ENVELOPE_VERSION,
        "alg": ALGORITHM,
        "kid": ctx.key_id,
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ct": base64.b64encode(ciphertext).decode("ascii"),
    }
    return json.dumps(envelope, separators=(",", ":"))


def _decrypt_raw(ctx: EncryptionContext, envelope_str: str, *, aad: bytes) -> str:
    require_cryptography()
    envelope = _parse_envelope(envelope_str)

    version = envelope.get("v")
    if version != ENVELOPE_VERSION:
        raise UnsupportedEnvelopeVersionError(
            f"Unsupported envelope version {version!r}; this build supports v{ENVELOPE_VERSION}."
        )
    if envelope.get("alg") != ALGORITHM:
        raise MalformedEnvelopeError(f"Unsupported or unknown algorithm: {envelope.get('alg')!r}")

    kid = envelope.get("kid")
    if kid != ctx.key_id:
        raise WrongKeyError(
            f"Envelope key id {kid!r} does not match the active key id {ctx.key_id!r}."
        )

    try:
        nonce = base64.b64decode(envelope["nonce"], validate=True)
        ciphertext = base64.b64decode(envelope["ct"], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MalformedEnvelopeError(f"Malformed base64 in envelope: {exc}") from exc

    aesgcm = AESGCM(ctx._enc_key)
    try:
        plaintext = aesgcm.decrypt(nonce, ciphertext, aad)
    except InvalidTag as exc:
        raise TamperError(
            "AEAD authentication failed with a matching key id; the ciphertext, "
            "nonce, or table/column/row binding was altered or corrupted."
        ) from exc
    return plaintext.decode("utf-8")


def encrypt_field(
    ctx: EncryptionContext,
    plaintext: str,
    *,
    table: str,
    column: str,
    row_identity: str,
) -> str:
    """Encrypt *plaintext* into a versioned AES-256-GCM JSON envelope string."""
    require_cryptography()
    aad = _aad(ctx, table=table, column=column, row_identity=row_identity)
    return _encrypt_raw(ctx, plaintext, aad=aad)


def encrypt_portable(
    ctx: EncryptionContext,
    plaintext: str,
    *,
    purpose: str,
    identity: str,
) -> str:
    """Encrypt a portable artifact (bundle/transfer) without binding to db_identity.

    AAD is ``memorymesh-portable|v1|{purpose}|{identity}`` so a bundle can be
    imported into another database that shares the same key material.
    """
    require_cryptography()
    aad = f"memorymesh-portable|v1|{purpose}|{identity}".encode("utf-8")
    return _encrypt_raw(ctx, plaintext, aad=aad)


def _parse_envelope(envelope_str: str) -> dict[str, Any]:
    if not isinstance(envelope_str, str):
        raise MalformedEnvelopeError(f"Envelope must be a string, got {type(envelope_str).__name__}.")
    try:
        data = json.loads(envelope_str)
    except (json.JSONDecodeError, ValueError) as exc:
        raise MalformedEnvelopeError(f"Envelope is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise MalformedEnvelopeError("Envelope JSON must be an object.")
    for required in ("v", "alg", "kid", "nonce", "ct"):
        if required not in data:
            raise MalformedEnvelopeError(f"Envelope missing required field: {required!r}")
    return data


def decrypt_field(
    ctx: EncryptionContext,
    envelope_str: str,
    *,
    table: str,
    column: str,
    row_identity: str,
) -> str:
    """Decrypt a versioned envelope string produced by :func:`encrypt_field`.

    Raises :class:`MalformedEnvelopeError` for structurally invalid input,
    :class:`UnsupportedEnvelopeVersionError` for an unknown ``v``,
    :class:`WrongKeyError` when the envelope's ``kid`` doesn't match *ctx*,
    and :class:`TamperError` when AEAD authentication fails despite a
    matching key id (ciphertext/AAD corruption or tampering).
    """
    aad = _aad(ctx, table=table, column=column, row_identity=row_identity)
    return _decrypt_raw(ctx, envelope_str, aad=aad)


def decrypt_portable(
    ctx: EncryptionContext,
    envelope_str: str,
    *,
    purpose: str,
    identity: str,
) -> str:
    """Decrypt a portable artifact produced by :func:`encrypt_portable`."""
    aad = f"memorymesh-portable|v1|{purpose}|{identity}".encode("utf-8")
    return _decrypt_raw(ctx, envelope_str, aad=aad)


def is_envelope(value: object) -> bool:
    """True if *value* structurally looks like an encryption envelope string.

    Used to detect ciphertext accidentally read as plaintext (e.g. a caller
    running in plaintext mode against a partially-encrypted database) so it
    is never silently returned as if it were real plaintext content.
    """
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return False
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    return all(k in data for k in ("v", "alg", "kid", "nonce", "ct"))


def content_fingerprint(
    ctx: EncryptionContext,
    *,
    project: str,
    provider: str,
    conversation_id: str,
    role: str,
    content: str,
) -> str:
    """Hex HMAC-SHA256 over the message namespace + content (dedupe/lookup aid).

    Uses the context's ``content_hmac_key`` so the fingerprint is stable for
    identical (namespace, content) pairs but not invertible without the key.
    """
    namespace = f"{project}|{provider}|{conversation_id}|{role}"
    message = f"{namespace}|{content}".encode("utf-8")
    return hmac.new(ctx.content_hmac_key, message, hashlib.sha256).hexdigest()


def keyed_term_token(ctx: EncryptionContext, plaintext_term: str) -> str:
    """Hex HMAC-SHA256 of a single lexical term for the keyed search index."""
    return hmac.new(ctx.term_hmac_key, plaintext_term.encode("utf-8"), hashlib.sha256).hexdigest()
