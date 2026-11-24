#!/usr/bin/env python3
"""Fail when the certificate a site serves is invalid or close to expiry.

certbot on the droplet owns TLS for virtio-accel.org and this repository
cannot renew anything. What it can do is notice: a renewal that silently
stopped working is invisible until the certificate expires, and from then on
every visitor gets a browser security error instead of the site. This script
is the check behind the scheduled ``Website TLS`` workflow and the post-deploy
verification in ``website.yml``.

For each ``host[:port]`` it opens one TLS connection with the platform trust
store, verifying the chain and the hostname, and reads ``notAfter`` from the
leaf the server presented. It looks at what is *served*, not at files on the
droplet, so a certificate that renewed on disk but that nginx never reloaded
is caught as well. Each host ends in exactly one state:

``ok``
    The chain verifies and enough validity is left.
``expiring``
    The chain verifies, but less validity is left than the threshold. certbot
    renews with a third of the lifetime left (30 days of a 90-day certificate),
    so dropping below the threshold means renewal is already broken.
``invalid``
    Verification failed: expired, wrong hostname, untrusted issuer. This is
    deterministic, so it is never retried and can never be downgraded.
``unreachable``
    No TLS handshake completed after every retry (DNS, TCP, timeout, or a
    protocol error), so nothing is known about the certificate.

The threshold is ``--min-days`` capped at a quarter of the certificate's own
lifetime. Let's Encrypt is shortening lifetimes (64 days from 2027-02-10, 45
days from 2028-02-16); a fixed 21 days would sit on top of certbot's renewal
point for a 64-day certificate and above it for a 45-day one, and a monitor
that fails during healthy operation gets ignored. A quarter of the lifetime
always falls after the renewal point and still leaves over a week to react.

Exit status: 0 when every host is ``ok`` (or its failure was downgraded with
``--warn``), 1 for a certificate problem, 2 when the check could not be made.

Usage::

    python3 website/tls_check.py virtio-accel.org
    python3 website/tls_check.py --warn expiring --warn unreachable virtio-accel.org

Standard library only, so it runs on a bare runner without a setup step.
"""

from __future__ import annotations

import argparse
import enum
import os
import re
import socket
import ssl
import string
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

DEFAULT_MIN_DAYS = 21.0
# The effective threshold never exceeds lifetime / LIFETIME_DIVISOR.
LIFETIME_DIVISOR = 4
ONE_DAY = timedelta(days=1)

# One DNS name: dot-separated LDH labels, 253 characters at most. Targets are
# echoed into workflow commands, so nothing else is accepted.
_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
_HOSTNAME = re.compile(rf"(?=.{{1,253}}\Z){_LABEL}(?:\.{_LABEL})*")

# Everything printed passes through this alphabet. Certificate fields and
# error strings come from the network; without `%`, CR, LF, `|`, `<` or a
# backtick they can neither forge a workflow command nor break the summary.
_PLAIN = frozenset(string.ascii_letters + string.digits + " .,:;=/'()[]_+-")


class Status(enum.Enum):
    OK = "ok"
    EXPIRING = "expiring"
    INVALID = "invalid"
    UNREACHABLE = "unreachable"


# The failures a caller may downgrade to a warning. INVALID is deliberately
# absent: a certificate browsers reject is never acceptable.
DOWNGRADABLE = (Status.EXPIRING, Status.UNREACHABLE)


@dataclass(frozen=True)
class Target:
    host: str
    port: int = 443

    def __str__(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass(frozen=True)
class Certificate:
    """The verified leaf a server presented."""

    not_before: datetime
    not_after: datetime
    issuer: str
    peer: str

    @property
    def lifetime(self) -> timedelta:
        return self.not_after - self.not_before


@dataclass(frozen=True)
class Result:
    target: Target
    status: Status
    detail: str


def parse_target(text: str) -> Target:
    """Parse ``host`` or ``host:port``; the port defaults to 443."""
    host, separator, port = text.partition(":")
    valid = bool(_HOSTNAME.fullmatch(host))
    if separator:
        valid = valid and port.isascii() and port.isdigit() and 0 < int(port) < 65536
    if not valid:
        raise argparse.ArgumentTypeError(
            f"expected a DNS hostname with an optional :port, got {text!r}"
        )
    return Target(host, int(port) if separator else 443)


def decode_certificate(certificate: dict, peer: str) -> Certificate:
    """Turn ``SSLSocket.getpeercert()`` output into a :class:`Certificate`."""

    def utc(field: str) -> datetime:
        seconds = ssl.cert_time_to_seconds(certificate[field])
        return datetime.fromtimestamp(seconds, timezone.utc)

    issuer = ", ".join(
        value for rdn in certificate.get("issuer", ()) for _, value in rdn
    )
    return Certificate(utc("notBefore"), utc("notAfter"), issuer, peer)


def fetch_certificate(
    target: Target, *, context: ssl.SSLContext, timeout: float
) -> Certificate:
    """Complete one verified handshake and return the leaf certificate.

    Raises ``ssl.SSLCertVerificationError`` when the chain or the hostname
    does not verify and another ``OSError`` when no handshake completes.
    """
    if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
        # getpeercert() returns nothing useful without verification, and a
        # monitor that does not verify would report a broken site as healthy.
        raise ValueError("the TLS context must verify the chain and the hostname")
    with socket.create_connection((target.host, target.port), timeout=timeout) as raw:
        with context.wrap_socket(raw, server_hostname=target.host) as tls:
            return decode_certificate(tls.getpeercert(), tls.getpeername()[0])


def expiry_threshold(certificate: Certificate, min_days: float) -> timedelta:
    """Validity below which renewal must already have happened."""
    return min(timedelta(days=min_days), certificate.lifetime / LIFETIME_DIVISOR)


def _days(span: timedelta) -> str:
    return f"{span / ONE_DAY:.1f} days"


def evaluate(
    target: Target, certificate: Certificate, now: datetime, min_days: float
) -> Result:
    """Classify a certificate that already verified."""
    remaining = certificate.not_after - now
    threshold = expiry_threshold(certificate, min_days)
    facts = (
        f"notAfter {certificate.not_after:%Y-%m-%d %H:%M:%S} UTC, "
        f"{_days(remaining)} left of a {_days(certificate.lifetime)} lifetime; "
        f"issuer {certificate.issuer}; peer {certificate.peer}"
    )
    if remaining < threshold:
        return Result(
            target,
            Status.EXPIRING,
            f"certificate is under the {_days(threshold)} renewal threshold, "
            f"so renewal is not happening ({facts})",
        )
    return Result(
        target,
        Status.OK,
        f"certificate verifies and clears the {_days(threshold)} threshold ({facts})",
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def check_target(
    target: Target,
    *,
    context: ssl.SSLContext,
    min_days: float = DEFAULT_MIN_DAYS,
    timeout: float = 20.0,
    attempts: int = 3,
    retry_delay: float = 5.0,
    now=_utc_now,
    sleep=time.sleep,
) -> Result:
    """Check one target, retrying only failures that can be transient."""
    failure: OSError | None = None
    for attempt in range(1, attempts + 1):
        try:
            certificate = fetch_certificate(target, context=context, timeout=timeout)
        except ssl.SSLCertVerificationError as error:
            return Result(
                target,
                Status.INVALID,
                f"certificate verification failed: {error.verify_message} "
                f"(OpenSSL verify code {error.verify_code})",
            )
        except OSError as error:
            failure = error
            if attempt < attempts:
                sleep(retry_delay * attempt)
        else:
            return evaluate(target, certificate, now(), min_days)
    return Result(
        target,
        Status.UNREACHABLE,
        f"no TLS handshake completed in {attempts} attempt(s); "
        f"last error: {type(failure).__name__}: {failure}",
    )


def _plain(text: str) -> str:
    return "".join(char if char in _PLAIN else "?" for char in text)[:600]


_TITLES = {
    Status.EXPIRING: "TLS certificate is not being renewed",
    Status.INVALID: "TLS certificate does not verify",
    Status.UNREACHABLE: "TLS certificate could not be checked",
}


def report(
    results: list[Result],
    *,
    warn: frozenset[Status] = frozenset(),
    environ=os.environ,
    stream=None,
) -> int:
    """Print the results, annotate failures, and return the exit status."""
    stream = sys.stdout if stream is None else stream
    actions = environ.get("GITHUB_ACTIONS") == "true"
    rows = []
    certificate_problem = unreachable = False
    for result in results:
        line = _plain(f"{result.target}: {result.detail}")
        if result.status is Status.OK:
            level = "ok"
            print(f"ok: {line}", file=stream)
        else:
            downgraded = result.status in warn and result.status in DOWNGRADABLE
            level = "warning" if downgraded else "error"
            if not downgraded and result.status is Status.UNREACHABLE:
                unreachable = True
            elif not downgraded:
                certificate_problem = True
            if actions:
                print(f"::{level} title={_TITLES[result.status]}::{line}", file=stream)
            else:
                print(f"{level}: {line}", file=stream)
        rows.append(f"| `{result.target}` | {level}: {result.status.value} | {line} |")
    if certificate_problem or unreachable:
        print(
            'Renewal runs on the droplet; see "TLS certificate" in website/README.md.',
            file=stream,
        )
    summary = environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as output:
            output.write("## TLS certificate\n\n| Host | Result | Detail |\n")
            output.write("| --- | --- | --- |\n" + "\n".join(rows) + "\n")
    if certificate_problem:
        return 1
    return 2 if unreachable else 0


def _non_negative(text: str) -> float:
    value = float(text)
    if not 0 <= value < float("inf"):
        raise argparse.ArgumentTypeError(f"must be a non-negative number: {text!r}")
    return value


def _positive(text: str) -> float:
    value = float(text)
    if not 0 < value < float("inf"):
        raise argparse.ArgumentTypeError(f"must be a positive number: {text!r}")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1: {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail when a served TLS certificate is invalid or close to expiry."
    )
    parser.add_argument(
        "targets", nargs="+", type=parse_target, metavar="host[:port]"
    )
    parser.add_argument(
        "--min-days",
        type=_non_negative,
        default=DEFAULT_MIN_DAYS,
        help="fail when fewer days of validity remain; capped at a quarter of "
        "the certificate's lifetime (default: %(default)s)",
    )
    parser.add_argument(
        "--warn",
        action="append",
        default=[],
        choices=[status.value for status in DOWNGRADABLE],
        help="report this outcome as a warning instead of failing; repeatable. "
        "A certificate that does not verify always fails.",
    )
    parser.add_argument(
        "--timeout",
        type=_positive,
        default=20.0,
        help="seconds allowed for each connection attempt (default: %(default)s)",
    )
    parser.add_argument(
        "--attempts",
        type=_positive_int,
        default=3,
        help="connection attempts before a host counts as unreachable "
        "(default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context = ssl.create_default_context()
    results = [
        check_target(
            target,
            context=context,
            min_days=args.min_days,
            timeout=args.timeout,
            attempts=args.attempts,
        )
        for target in args.targets
    ]
    return report(results, warn=frozenset(Status(name) for name in args.warn))


if __name__ == "__main__":
    sys.exit(main())
