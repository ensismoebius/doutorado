from experiment_microscope.core.bookmarks import Bookmark, BookmarkStore, make_bookmark


def test_bookmark_json_roundtrip():
    b = make_bookmark(
        "SNN latent collapse",
        {"experiment": "thesis", "window": 3},
        ["Thesis", "Phase 00 — feature ranking", "e05_p00_hc..."],
        "Triangle",
        b"\x01\x02\x03",
    )
    d = b.to_json()
    b2 = Bookmark.from_json(d)
    assert b2.name == b.name
    assert b2.selection == b.selection
    assert b2.nav_path == b.nav_path
    assert b2.active_tab == "Triangle"
    assert b2.window_state() == b"\x01\x02\x03"


def test_store_add_dedupes_by_name(qapp, tmp_path, monkeypatch):
    # isolate QSettings to a temp ini
    from PySide6.QtCore import QSettings

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    monkeypatch.setenv("HOME", str(tmp_path))
    store = BookmarkStore("test-org", "test-app-bookmarks")
    store.save([])
    store.add(make_bookmark("a", {"x": 1}, [], "Signal", b""))
    store.add(make_bookmark("a", {"x": 2}, [], "Signal", b""))  # same name
    names = [b.name for b in store.load()]
    assert names.count("a") == 1
    assert store.load()[0].selection == {"x": 2}


def test_workspace_save_and_list_bookmark(qapp):
    from experiment_microscope.app.workspace import Workspace

    w = Workspace()
    try:
        w.bookmarks.store.save([])  # clean slate
        w._save_bookmark("test-bm")
        names = [w.bookmarks._list.item(i).text() for i in range(w.bookmarks._list.count())]
        assert "test-bm" in names
        w.bookmarks.store.remove("test-bm")
    finally:
        w.close()
