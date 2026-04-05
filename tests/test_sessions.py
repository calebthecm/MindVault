"""Tests for session manager — pure file I/O, no LLM."""

import gzip
import time

import pytest

from mindvault.sessions.manager import Session, load_session, load_last_session, list_sessions


@pytest.fixture
def sessions_dir(tmp_path):
    return tmp_path / "sessions"


def test_session_save_and_load(sessions_dir):
    session = Session(sessions_dir, model="llama3.2")
    session.add_turn("user", "What projects am I working on?")
    session.add_turn("assistant", "You are working on MindVault.")
    path = session.save_and_index()

    assert path.exists()
    loaded = load_session(sessions_dir, session.session_id)
    assert loaded is not None
    assert len(loaded.turns) == 2
    assert loaded.turns[0]["role"] == "user"
    assert loaded.model == "llama3.2"


def test_session_is_compressed(sessions_dir):
    session = Session(sessions_dir, model="llama3.2")
    session.add_turn("user", "hello")
    path = session.save()
    data = gzip.decompress(path.read_bytes())
    assert b"session_id" in data


def test_load_last_session(sessions_dir):
    s1 = Session(sessions_dir, model="llama3.2")
    s1.add_turn("user", "first")
    s1.save_and_index()

    time.sleep(1.1)  # ensure different timestamp in session_id

    s2 = Session(sessions_dir, model="llama3.2")
    s2.add_turn("user", "second")
    s2.save_and_index()

    last = load_last_session(sessions_dir)
    assert last is not None
    assert last.session_id == s2.session_id


def test_list_sessions_sorted(sessions_dir):
    for i in range(3):
        s = Session(sessions_dir, model="llama3.2")
        s.add_turn("user", f"message {i}")
        s.save_and_index()
        time.sleep(1.1)

    sessions = list_sessions(sessions_dir)
    assert len(sessions) == 3
    dates = [s["started_at"] for s in sessions]
    assert dates == sorted(dates, reverse=True)


def test_session_status_raw_by_default(sessions_dir):
    session = Session(sessions_dir, model="llama3.2")
    assert session.status == "raw"


def test_resume_restores_turns(sessions_dir):
    session = Session(sessions_dir, model="llama3.2")
    session.add_turn("user", "I want to build an agency")
    session.add_turn("assistant", "Great idea!")
    session.summary = "Discussed building an agency."
    session.status = "processed"
    session.save_and_index()

    loaded = load_session(sessions_dir, session.session_id)
    assert loaded.summary == "Discussed building an agency."
    assert loaded.status == "processed"
    assert len(loaded.turns) == 2
