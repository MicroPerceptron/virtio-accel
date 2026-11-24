"""Checks for the served-certificate monitor.

The policy tests use synthetic dates. The handshake tests mint real
certificates from a throwaway CA and serve them from a loopback TLS endpoint,
so chain, hostname and validity-window verification are exercised through the
same code path the scheduled workflow uses against virtio-accel.org.
"""
import argparse
import contextlib
import io
import socket
import socketserver
import ssl
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

import tls_check
from tls_check import Certificate, Result, Status, Target

DAY = timedelta(days=1)
NOW = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)


def certificate(lifetime_days, days_left, now=NOW):
    not_after = now + days_left * DAY
    return Certificate(not_after - lifetime_days * DAY, not_after, 'Test CA', '192.0.2.1')


class ThresholdTests(unittest.TestCase):
    def status(self, lifetime_days, days_left, min_days=tls_check.DEFAULT_MIN_DAYS):
        return tls_check.evaluate(
            Target('example.org'), certificate(lifetime_days, days_left), NOW, min_days).status

    def test_ninety_day_certificate_fails_under_twenty_one_days(self):
        self.assertEqual(
            tls_check.expiry_threshold(certificate(90, 60), 21), 21 * DAY)
        self.assertIs(self.status(90, 60), Status.OK)
        self.assertIs(self.status(90, 30), Status.OK)
        # "Fewer than" the threshold: exactly 21 days left still passes.
        self.assertIs(self.status(90, 21), Status.OK)
        self.assertIs(self.status(90, 20.99), Status.EXPIRING)
        self.assertIs(self.status(90, 1), Status.EXPIRING)

    def test_a_verified_certificate_past_not_after_is_still_a_failure(self):
        self.assertIs(self.status(90, -1), Status.EXPIRING)

    def test_shorter_lifetimes_do_not_fail_before_certbot_would_renew(self):
        # certbot renews with a third of the lifetime left; the threshold is a
        # quarter, so a healthy renewal cadence never trips it.
        for lifetime in (64, 45, 6):
            with self.subTest(lifetime=lifetime):
                renewal_point = lifetime / 3
                self.assertEqual(
                    tls_check.expiry_threshold(certificate(lifetime, lifetime), 21),
                    lifetime * DAY / 4)
                self.assertIs(self.status(lifetime, renewal_point), Status.OK)
                self.assertIs(self.status(lifetime, lifetime / 4), Status.OK)
                self.assertIs(self.status(lifetime, lifetime / 4 - 0.01), Status.EXPIRING)

    def test_min_days_lowers_but_never_raises_the_lifetime_cap(self):
        self.assertIs(self.status(90, 10, min_days=7), Status.OK)
        self.assertIs(self.status(90, 6, min_days=7), Status.EXPIRING)
        self.assertIs(self.status(90, 23, min_days=60), Status.OK)
        self.assertIs(self.status(90, 22, min_days=60), Status.EXPIRING)
        self.assertIs(self.status(90, 0.01, min_days=0), Status.OK)

    def test_detail_reports_expiry_and_threshold(self):
        detail = tls_check.evaluate(
            Target('example.org'), certificate(90, 12), NOW, 21).detail
        self.assertIn('notAfter 2026-12-05 12:00:00 UTC', detail)
        self.assertIn('12.0 days left', detail)
        self.assertIn('21.0 days renewal threshold', detail)


class DecodingTests(unittest.TestCase):
    def test_peer_certificate_dictionary(self):
        decoded = tls_check.decode_certificate({
            'notBefore': 'Aug 24 20:47:10 2026 GMT',
            'notAfter': 'Nov 22 20:47:09 2026 GMT',
            'issuer': ((('countryName', 'US'),), (('organizationName', "Let's Encrypt"),),
                       (('commonName', 'YE1'),)),
        }, '192.0.2.1')
        self.assertEqual(decoded.not_before, datetime(2026, 8, 24, 20, 47, 10, tzinfo=timezone.utc))
        self.assertEqual(decoded.not_after, datetime(2026, 11, 22, 20, 47, 9, tzinfo=timezone.utc))
        self.assertEqual(decoded.issuer, "US, Let's Encrypt, YE1")
        self.assertEqual(decoded.lifetime, 90 * DAY - timedelta(seconds=1))

    def test_targets(self):
        self.assertEqual(tls_check.parse_target('virtio-accel.org'), Target('virtio-accel.org', 443))
        self.assertEqual(tls_check.parse_target('localhost:8443'), Target('localhost', 8443))
        self.assertEqual(str(Target('virtio-accel.org')), 'virtio-accel.org:443')
        for text in ('', ':443', 'https://virtio-accel.org', 'virtio-accel.org/', 'a..b',
                     '-a.org', 'a.org:0', 'a.org:65536', 'a.org:44x', 'a.org:', 'a b',
                     'a.org\n::error::x', '[::1]:443', 'x' * 64 + '.org'):
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                tls_check.parse_target(text)


class ReportTests(unittest.TestCase):
    def report(self, results, warn=(), actions=True):
        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / 'summary.md'
            environ = {'GITHUB_STEP_SUMMARY': str(summary)}
            if actions:
                environ['GITHUB_ACTIONS'] = 'true'
            code = tls_check.report(
                results, warn=frozenset(warn), environ=environ, stream=stream)
            return code, stream.getvalue(), summary.read_text()

    def result(self, status, detail='detail'):
        return Result(Target('example.org'), status, detail)

    def test_exit_status_and_annotations(self):
        for status, expected in ((Status.OK, 0), (Status.EXPIRING, 1),
                                 (Status.INVALID, 1), (Status.UNREACHABLE, 2)):
            with self.subTest(status=status):
                code, output, summary = self.report([self.result(status)])
                self.assertEqual(code, expected)
                self.assertEqual('::error title=' in output, expected != 0)
                self.assertIn(status.value, summary)

    def test_certificate_problems_outrank_unreachable_hosts(self):
        code, _, _ = self.report(
            [self.result(Status.UNREACHABLE), self.result(Status.EXPIRING)])
        self.assertEqual(code, 1)

    def test_expiring_and_unreachable_can_be_downgraded_to_warnings(self):
        for status in (Status.EXPIRING, Status.UNREACHABLE):
            with self.subTest(status=status):
                code, output, _ = self.report([self.result(status)], warn=[status])
                self.assertEqual(code, 0)
                self.assertIn('::warning title=', output)
                self.assertNotIn('::error', output)

    def test_verification_failure_can_never_be_downgraded(self):
        code, output, _ = self.report([self.result(Status.INVALID)], warn=list(Status))
        self.assertEqual(code, 1)
        self.assertIn('::error title=TLS certificate does not verify::', output)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            tls_check.build_parser().parse_args(['--warn', 'invalid', 'example.org'])

    def test_network_supplied_text_cannot_forge_commands_or_markup(self):
        hostile = 'x\n::error::forged\r%0A::set-output name=a::b | <b>`code`'
        code, output, summary = self.report([self.result(Status.OK, hostile)])
        self.assertEqual(code, 0)
        self.assertEqual(len(output.splitlines()), 1)
        self.assertTrue(output.startswith('ok: example.org:443: x?::error::forged'))
        for forbidden in '%|<>`\r':
            self.assertNotIn(forbidden, output)
        # The row keeps its three cells and its only code span is the host.
        row = summary.splitlines()[-1]
        self.assertEqual((row.count('|'), row.count('`'), row.count('<')), (4, 2, 0))

    def test_nonsensical_options_are_rejected(self):
        for arguments in (['--min-days', '-1', 'example.org'], ['--min-days', 'nan', 'example.org'],
                          ['--timeout', '0', 'example.org'], ['--attempts', '0', 'example.org'],
                          ['--warn', 'ok', 'example.org'], []):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit), \
                    contextlib.redirect_stderr(io.StringIO()):
                tls_check.build_parser().parse_args(arguments)
        options = tls_check.build_parser().parse_args(['example.org'])
        self.assertEqual((options.min_days, options.warn, options.attempts), (21.0, [], 3))

    def test_plain_output_outside_actions(self):
        code, output, _ = self.report([self.result(Status.INVALID)], actions=False)
        self.assertEqual(code, 1)
        self.assertTrue(output.startswith('error: example.org:443: detail'))
        self.assertNotIn('::', output)


class Authority:
    """A throwaway CA that can pass OpenSSL's strict X.509 checks."""

    def __init__(self, directory, name):
        self.directory = Path(directory)
        self.key = ec.generate_private_key(ec.SECP256R1())
        self.name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
        now = datetime.now(timezone.utc)
        self.certificate = (
            x509.CertificateBuilder()
            .subject_name(self.name).issuer_name(self.name)
            .public_key(self.key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - 400 * DAY).not_valid_after(now + 400 * DAY)
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.KeyUsage(
                digital_signature=False, content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False, key_cert_sign=True,
                crl_sign=True, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(
                x509.SubjectKeyIdentifier.from_public_key(self.key.public_key()), critical=False)
            .sign(self.key, hashes.SHA256()))
        self.path = self.directory / f'{name}.pem'
        self.path.write_bytes(self.certificate.public_bytes(serialization.Encoding.PEM))

    def trust(self):
        return ssl.create_default_context(cafile=str(self.path))

    def issue(self, label, *, days_before, days_after, hostname='localhost'):
        """Write a leaf valid from ``days_before`` ago until ``days_after`` from now."""
        key = ec.generate_private_key(ec.SECP256R1())
        now = datetime.now(timezone.utc)
        leaf = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)]))
            .issuer_name(self.name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - days_before * DAY).not_valid_after(now + days_after * DAY)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(
                digital_signature=True, content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False, key_cert_sign=False,
                crl_sign=False, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(
                x509.SubjectAlternativeName([x509.DNSName(hostname)]), critical=False)
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(self.key.public_key()),
                critical=False)
            .sign(self.key, hashes.SHA256()))
        path = self.directory / f'{label}.pem'
        path.write_bytes(
            leaf.public_bytes(serialization.Encoding.PEM)
            + key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption()))
        return path


class _Handshake(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(5)
        try:
            if self.server.tls is None:
                self.request.sendall(b'HTTP/1.1 400 Bad Request\r\n\r\n')
            else:
                # A client that rejects the certificate aborts the handshake.
                self.server.tls.wrap_socket(self.request, server_side=True).close()
        except OSError:
            pass


@contextlib.contextmanager
def serving(chain=None):
    """Serve ``chain`` over TLS on loopback, or plain text when it is None."""
    server = socketserver.TCPServer(('127.0.0.1', 0), _Handshake)
    server.tls = None
    if chain is not None:
        server.tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server.tls.load_cert_chain(chain)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01})
    thread.start()
    try:
        yield Target('localhost', server.server_address[1])
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def closed_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return Target('localhost', listener.getsockname()[1])


class HandshakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.authority = Authority(directory.name, 'trusted')
        cls.stranger = Authority(directory.name, 'stranger')
        issue = cls.authority.issue
        cls.healthy = issue('healthy', days_before=30, days_after=60)
        cls.expiring = issue('expiring', days_before=80, days_after=10)
        cls.short_lived = issue('short', days_before=33, days_after=12)
        cls.expired = issue('expired', days_before=91, days_after=-1)
        cls.premature = issue('premature', days_before=-1, days_after=91)
        cls.wrong_host = issue(
            'wrong-host', days_before=30, days_after=60, hostname='elsewhere.invalid')
        cls.untrusted = cls.stranger.issue('untrusted', days_before=30, days_after=60)

    def check(self, chain, **options):
        sleeps = []
        options.setdefault('context', self.authority.trust())
        with serving(chain) as target:
            result = tls_check.check_target(
                target, timeout=5, sleep=sleeps.append, **options)
        return result, sleeps

    def test_healthy_certificate_passes(self):
        result, sleeps = self.check(self.healthy)
        self.assertIs(result.status, Status.OK, result.detail)
        self.assertIn('60.0 days left of a 90.0 days lifetime', result.detail)
        self.assertIn('issuer trusted; peer 127.0.0.1', result.detail)
        self.assertEqual(sleeps, [])

    def test_certificate_with_ten_days_left_fails(self):
        result, sleeps = self.check(self.expiring)
        self.assertIs(result.status, Status.EXPIRING, result.detail)
        self.assertIn('10.0 days left', result.detail)
        self.assertIn('under the 21.0 days renewal threshold', result.detail)
        self.assertEqual(sleeps, [])

    def test_forty_five_day_certificate_with_twelve_days_left_passes(self):
        result, _ = self.check(self.short_lived)
        self.assertIs(result.status, Status.OK, result.detail)
        self.assertIn('clears the 11.2 days threshold', result.detail)

    def test_verification_failures_are_invalid_and_never_retried(self):
        for chain, message in ((self.expired, 'certificate has expired'),
                               (self.premature, 'certificate is not yet valid'),
                               (self.wrong_host, 'Hostname mismatch'),
                               (self.untrusted, 'unable to get local issuer certificate')):
            with self.subTest(chain=chain.name):
                result, sleeps = self.check(chain)
                self.assertIs(result.status, Status.INVALID, result.detail)
                self.assertIn(message, result.detail)
                self.assertEqual(sleeps, [])

    def test_platform_trust_store_rejects_the_private_authority(self):
        result, _ = self.check(self.healthy, context=ssl.create_default_context())
        self.assertIs(result.status, Status.INVALID, result.detail)

    def test_certificate_must_name_the_host_that_was_asked_for(self):
        # Same endpoint and certificate as the healthy case, reached under a
        # name the certificate does not carry.
        with serving(self.healthy) as target:
            result = tls_check.check_target(
                Target('127.0.0.1', target.port), context=self.authority.trust(), timeout=5)
        self.assertIs(result.status, Status.INVALID, result.detail)
        self.assertIn('IP address mismatch', result.detail)

    def test_context_that_skips_verification_is_refused(self):
        def no_hostname(context):
            context.check_hostname = False

        def no_chain(context):
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

        def optional_chain(context):
            context.verify_mode = ssl.CERT_OPTIONAL

        for weaken in (no_hostname, no_chain, optional_chain):
            context = self.authority.trust()
            weaken(context)
            with self.subTest(weaken=weaken.__name__), self.assertRaises(ValueError):
                tls_check.check_target(closed_port(), context=context)

    def test_refused_connection_is_unreachable_after_every_attempt(self):
        sleeps = []
        result = tls_check.check_target(
            closed_port(), context=self.authority.trust(), timeout=5, attempts=3,
            retry_delay=5, sleep=sleeps.append)
        self.assertIs(result.status, Status.UNREACHABLE)
        self.assertIn('in 3 attempt(s)', result.detail)
        self.assertIn('ConnectionRefusedError', result.detail)
        self.assertEqual(sleeps, [5, 10])

    def test_endpoint_that_does_not_speak_tls_is_unreachable(self):
        result, sleeps = self.check(None, attempts=2)
        self.assertIs(result.status, Status.UNREACHABLE, result.detail)
        self.assertEqual(len(sleeps), 1)

    def test_stalled_handshake_times_out(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen()
            result = tls_check.check_target(
                Target('localhost', listener.getsockname()[1]),
                context=self.authority.trust(), timeout=0.2, attempts=1)
        self.assertIs(result.status, Status.UNREACHABLE)
        self.assertIn('timed out', result.detail)

    def test_transient_failure_is_retried(self):
        sleeps = []
        with serving(self.healthy) as target:
            fetch = tls_check.fetch_certificate
            with patch.object(tls_check, 'fetch_certificate', side_effect=[
                    ConnectionResetError('reset'),
                    fetch(target, context=self.authority.trust(), timeout=5)]):
                result = tls_check.check_target(
                    target, context=self.authority.trust(), sleep=sleeps.append)
        self.assertIs(result.status, Status.OK, result.detail)
        self.assertEqual(sleeps, [5])

    def main(self, chain, *arguments):
        """Run the command line as a workflow step would, with our CA trusted."""
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, serving(chain) as target:
            summary = Path(directory) / 'summary.md'
            environ = {'GITHUB_ACTIONS': 'true', 'GITHUB_STEP_SUMMARY': str(summary)}
            with patch.object(tls_check.ssl, 'create_default_context',
                              return_value=self.authority.trust()), \
                    patch.dict(tls_check.os.environ, environ), \
                    contextlib.redirect_stdout(output):
                code = tls_check.main([*arguments, '--attempts', '1', str(target)])
            return code, output.getvalue(), summary.read_text()

    def test_command_line_passes_on_a_healthy_certificate(self):
        code, output, summary = self.main(self.healthy)
        self.assertEqual(code, 0)
        self.assertTrue(output.startswith('ok: localhost:'))
        self.assertNotIn('::', output)
        self.assertIn('| ok: ok |', summary)

    def test_command_line_fails_under_the_threshold(self):
        code, output, summary = self.main(self.expiring)
        self.assertEqual(code, 1)
        self.assertIn('::error title=TLS certificate is not being renewed::', output)
        self.assertIn('| error: expiring |', summary)
        # The post-deploy verification leaves that failure to the monitor...
        code, output, _ = self.main(self.expiring, '--warn', 'expiring', '--warn', 'unreachable')
        self.assertEqual(code, 0)
        self.assertIn('::warning title=TLS certificate is not being renewed::', output)

    def test_command_line_always_fails_on_an_expired_certificate(self):
        # ...but a certificate browsers reject fails there as well.
        for arguments in ((), ('--warn', 'expiring', '--warn', 'unreachable')):
            with self.subTest(arguments=arguments):
                code, output, summary = self.main(self.expired, *arguments)
                self.assertEqual(code, 1)
                self.assertIn('::error title=TLS certificate does not verify::', output)
                self.assertIn('certificate has expired', output)
                self.assertIn('| error: invalid |', summary)

    def test_command_line_reports_unreachable_hosts(self):
        code, output, _ = self.main(None)
        self.assertEqual(code, 2)
        self.assertIn('::error title=TLS certificate could not be checked::', output)
        code, output, _ = self.main(None, '--warn', 'unreachable')
        self.assertEqual(code, 0)
        self.assertIn('::warning title=TLS certificate could not be checked::', output)


if __name__ == '__main__':
    unittest.main()
