from privatemode.document_store.app import vector_db_config
from privatemode.document_store.auth import set_auth_key_path

def setup_test(tmp_path):
    db_path = tmp_path / "data/test_chroma"
    vector_db_config["local_embedding"] = True
    vector_db_config["path"] = db_path
    set_auth_key_path(db_path)

    # delete any existing test db
    import shutil, os
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
