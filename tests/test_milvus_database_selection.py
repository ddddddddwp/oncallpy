from app.rag import milvus_client


def test_native_milvus_clients_select_configured_database(monkeypatch):
    connection_calls = []
    client_calls = []

    def record_connection(**kwargs):
        connection_calls.append(kwargs)

    class RecordingClient:
        def __init__(self, **kwargs):
            client_calls.append(kwargs)

    monkeypatch.setattr(milvus_client.config, "milvus_database_name", "oncall_agent")
    monkeypatch.setattr(milvus_client.connections, "connect", record_connection)
    monkeypatch.setattr(milvus_client, "MilvusClient", RecordingClient)

    manager = milvus_client.MilvusClientManager()
    monkeypatch.setattr(manager, "_collection_exists", lambda: False)
    monkeypatch.setattr(manager, "_create_collection", lambda: None)
    monkeypatch.setattr(manager, "_load_collection", lambda: None)

    manager.connect()

    assert connection_calls[0]["db_name"] == "oncall_agent"
    assert client_calls[0]["db_name"] == "oncall_agent"
