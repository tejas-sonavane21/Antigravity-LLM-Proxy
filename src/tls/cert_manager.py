"""
src/tls/cert_manager.py — TLS Certificate Manager
====================================================
Generates and manages the self-signed certificate used by the HTTPS proxy server.
The IDE trusts our proxy via NODE_TLS_REJECT_UNAUTHORIZED='0' (injected by patcher).

Certificate spec (matched to reference cert.rs L42-L67):
  - RSA 2048-bit key
  - CN = "AG Proxy Local CA", O = "AG Proxy Manager"
  - SAN: DNS:localhost, IP:127.0.0.1
  - BasicConstraints: CA=true
  - Validity: 10 years
  - Format: PEM (both cert and key)

Reference: cert.rs (full file) - certificate generation logic
Reference: Brainstorming Section 9.2 - TLS/HTTPS handling explanation
"""

import datetime
import ipaddress
import logging
import os
import ssl
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from src.config import TlsConfig


class CertManager:
    """
    Manages the self-signed TLS certificate for the proxy HTTPS server.

    Usage:
        cert_manager = CertManager(config.tls)
        cert_path, key_path = cert_manager.ensure_certs()
        ssl_ctx = cert_manager.get_ssl_context()
    """

    def __init__(self, tls_config: TlsConfig) -> None:
        self.config = tls_config
        self.logger = logging.getLogger("tls")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def ensure_certs(self) -> tuple:
        """
        Ensure that a valid cert + key pair exists on disk.

        - If both files exist: log and return existing paths.
        - If either is missing: generate a new pair, save to disk, return paths.
        - Creates cert_dir automatically if it doesn't exist.

        Returns:
            tuple: (cert_path: str, key_path: str) — absolute paths to PEM files.
        """
        cert_path = self.config.cert_path
        key_path = self.config.key_path

        # Auto-create the certificate directory if it doesn't exist
        os.makedirs(self.config.cert_dir, exist_ok=True)
        self.logger.debug(f"Certificate directory: {self.config.cert_dir}")

        # Check if both files already exist
        if Path(cert_path).exists() and Path(key_path).exists():
            self.logger.info(f"Using existing certificate: {cert_path}")
            return cert_path, key_path

        # Either missing — regenerate both to keep the pair in sync
        if Path(cert_path).exists() and not Path(key_path).exists():
            self.logger.warning("Certificate exists but key is missing — regenerating pair")
        elif not Path(cert_path).exists() and Path(key_path).exists():
            self.logger.warning("Key exists but certificate is missing — regenerating pair")
        else:
            self.logger.info("No certificate found — generating new self-signed certificate")

        self._generate_self_signed_cert(cert_path, key_path)
        self.logger.info(f"Generated new self-signed certificate: {cert_path}")
        return cert_path, key_path

    def get_ssl_context(self) -> ssl.SSLContext:
        """
        Create and return an SSL context for the uvicorn HTTPS server.

        Must be called AFTER ensure_certs() — cert and key files must exist.

        Returns:
            ssl.SSLContext: Ready-to-use server-side SSL context.

        Raises:
            ssl.SSLError: If cert/key files are missing or corrupt.
        """
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(
            certfile=self.config.cert_path,
            keyfile=self.config.key_path,
        )
        self.logger.debug(f"SSL context loaded: {self.config.cert_path}")
        return ctx

    # -------------------------------------------------------------------------
    # Private — Certificate Generation
    # -------------------------------------------------------------------------

    def _generate_self_signed_cert(self, cert_path: str, key_path: str) -> None:
        """
        Generate an RSA 2048-bit self-signed CA certificate and write PEM files.

        Certificate parameters are matched 1:1 to the reference project's cert.rs:
          - cert.rs L48  → CN = "AG Proxy Local CA"
          - cert.rs L51  → O  = "AG Proxy Manager"
          - cert.rs L52  → BasicConstraints: CA=true (IsCa::Ca)
          - cert.rs L54  → SAN: DNS:localhost
          - cert.rs L55  → SAN: IP:127.0.0.1
          - cert.rs L58-59 → validity: ~10 years
          - cert.rs L61  → RSA 2048 key pair (KeyPair::generate())
          - cert.rs L66-67 → PEM output for cert and key

        Args:
            cert_path: Absolute path where the PEM certificate will be written.
            key_path:  Absolute path where the PEM private key will be written.

        Raises:
            RuntimeError: If key generation or certificate signing fails.
            OSError: If the files cannot be written to disk.
        """
        self.logger.debug("Generating RSA 2048-bit private key...")

        try:
            # Step 1: Generate RSA 2048-bit private key
            # cert.rs L61: KeyPair::generate() (defaults to RSA 2048)
            key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )

            # Step 2: Build subject/issuer name
            # cert.rs L47-L51: distinguished_name with CommonName + OrganizationName
            # Self-signed → subject == issuer
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, "AG Proxy Local CA"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AG Proxy Manager"),
            ])

            # Step 3: Determine validity window (10 years)
            # cert.rs L58-L59: not_before=2024-01-01, not_after=2034-12-31
            # We use current time as start for maximum compatibility
            now = datetime.datetime.utcnow()
            not_before = now
            not_after = now + datetime.timedelta(days=3650)  # ~10 years

            # Step 4: Build the certificate
            self.logger.debug("Building X.509 certificate structure...")
            cert = (
                x509.CertificateBuilder()
                .subject_name(subject)
                .issuer_name(issuer)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(not_before)
                .not_valid_after(not_after)
                # cert.rs L53-L56: SAN with localhost (DNS) and 127.0.0.1 (IP)
                .add_extension(
                    x509.SubjectAlternativeName([
                        x509.DNSName("localhost"),
                        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                    ]),
                    critical=False,
                )
                # cert.rs L52: IsCa::Ca(BasicConstraints::Unconstrained)
                .add_extension(
                    x509.BasicConstraints(ca=True, path_length=None),
                    critical=True,
                )
                .sign(key, hashes.SHA256())
            )

        except Exception as e:
            raise RuntimeError(f"Certificate generation failed: {e}") from e

        # Step 5: Write PEM files to disk
        # cert.rs L69-L70: fs::write for cert and key
        self.logger.debug(f"Writing certificate to: {cert_path}")
        try:
            with open(cert_path, "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
        except OSError as e:
            raise OSError(f"Failed to write certificate to {cert_path}: {e}") from e

        self.logger.debug(f"Writing private key to: {key_path}")
        try:
            with open(key_path, "wb") as f:
                f.write(key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    # No password — key is local-only, accessed only by our proxy
                    encryption_algorithm=serialization.NoEncryption(),
                ))
        except OSError as e:
            raise OSError(f"Failed to write private key to {key_path}: {e}") from e

        self.logger.debug("Certificate and key written successfully")
