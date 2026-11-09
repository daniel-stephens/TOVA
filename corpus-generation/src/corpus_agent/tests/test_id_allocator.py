import pytest

from corpus_state.db import create_database, create_instance

from corpus_agent.id_allocator import InstanceIdAllocator


class TestInstanceIdAllocator:
    def test_allocate_from_empty_db(self):
        conn = create_database(":memory:")
        alloc = InstanceIdAllocator(conn, "test-corpus", count=3)
        ids = [alloc.next_id() for _ in range(3)]
        assert ids == [1, 2, 3]

    def test_allocate_skips_existing_ids(self):
        conn = create_database(":memory:")
        create_instance(conn, "test-corpus", 1)
        create_instance(conn, "test-corpus", 2)
        alloc = InstanceIdAllocator(conn, "test-corpus", count=2)
        ids = [alloc.next_id() for _ in range(2)]
        assert ids == [3, 4]

    def test_exhausted_raises(self):
        conn = create_database(":memory:")
        alloc = InstanceIdAllocator(conn, "test-corpus", count=1)
        alloc.next_id()
        with pytest.raises(StopIteration):
            alloc.next_id()

    def test_allocate_does_not_collide_across_corpora(self):
        conn = create_database(":memory:")
        create_instance(conn, "corpus-a", 1)
        create_instance(conn, "corpus-a", 2)
        # second corpus must start after the global max, not the per-corpus max
        alloc = InstanceIdAllocator(conn, "corpus-b", count=2)
        ids = [alloc.next_id() for _ in range(2)]
        assert ids == [3, 4]

