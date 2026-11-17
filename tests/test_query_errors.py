from app.rag.query.errors import safe_error_summary


def test_safe_error_summary_handles_unprintable_values() -> None:
    class BadString:
        def __str__(self):
            raise RuntimeError("password=leak")

    assert safe_error_summary(BadString()) == "BadString"


def test_safe_error_summary_redacts_quoted_and_authorization_values() -> None:
    value = (
        '{"password": "hunter2", "api_key": "abc"} '
        'password="hunter two" Authorization: Basic dXNlcjpwYXNz '
        "Authorization=Bearer token-value"
    )
    summary = safe_error_summary(value)
    assert "hunter" not in summary
    assert "abc" not in summary
    assert "dXNlcjpwYXNz" not in summary
    assert "token-value" not in summary


def test_redaction_generalizes_prefixed_keys_and_preserves_adjacent_json() -> None:
    value = (
        '{"refresh_token":"r1","keep":"visible","db_password":"p1",'
        '"client_secret":"s1","my_api_key":"k1",'
        '"Authorization":"Unknown scheme and value","after":42,'
        '"Proxy-Authorization":"ApiKey proxy-value"} '
        "refresh_token='refresh with spaces' db_password=plain "
        "Authorization: Token header-value; keep_header=visible"
    )

    summary = safe_error_summary(value)

    for secret in (
        "r1",
        "p1",
        "s1",
        "k1",
        "scheme and value",
        "proxy-value",
        "refresh with spaces",
        "plain",
        "header-value",
    ):
        assert secret not in summary
    assert '"keep":"visible"' in summary
    assert '"after":42' in summary


def test_authorization_headers_redact_the_full_line_only() -> None:
    value = (
        "Authorization: AWS4-HMAC-SHA256 Credential=AKIA/20260723/region/service/aws4_request, "
        "SignedHeaders=host;x-amz-date, Signature=deadbeef\n"
        "ordinary message remains\n"
        'Proxy-Authorization: Digest username="user", realm="private", nonce="secret"\r\n'
        '{"Authorization":"Digest json-secret, qop=auth","keep":"visible"}'
    )

    summary = safe_error_summary(value)

    for secret in ("AKIA", "deadbeef", "username", "private", "nonce", "json-secret"):
        assert secret not in summary
    assert "ordinary message remains" in summary
    assert '"keep":"visible"' in summary
