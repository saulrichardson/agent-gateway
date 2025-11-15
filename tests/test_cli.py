from gateway.__main__ import main


def test_main_invokes_uvicorn(monkeypatch):
    called = {}

    def fake_run(*args, host, port, reload, **kwargs):
        called["host"] = host
        called["port"] = port
        called["reload"] = reload

    monkeypatch.setattr("gateway.__main__.uvicorn.run", fake_run)

    assert main(["--host", "127.0.0.1", "--port", "9001", "--reload"]) == 0
    assert called == {"host": "127.0.0.1", "port": 9001, "reload": True}
