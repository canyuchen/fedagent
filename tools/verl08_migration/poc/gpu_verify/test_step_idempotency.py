"""CPU test of the /step idempotency pattern used by both env services.

Replicates the _Session holder + /step handler logic VERBATIM (the servers can't be
imported here — they pull gym/alfworld) and asserts the four failure modes from the design:
  1. normal monotone sequence -> env steps exactly once per id
  2. replay (re-sent latest id) -> cached response, env NOT re-stepped
  3. out-of-order id -> 409 (loud, not silently applied)
  4. concurrent duplicate (retry races the original) -> env steps once, both get same resp
  5. /reset restarts the counter
  6. legacy id=-1 path -> applies every call (back-compat, no cache)
"""
import asyncio


class _Session:                       # VERBATIM copy of the server class
    __slots__ = ("env", "lock", "next_step_id", "last_step_id", "last_resp")

    def __init__(self, env):
        self.env = env
        self.lock = asyncio.Lock()
        self.next_step_id = 0
        self.last_step_id = -1
        self.last_resp = None

    def reset_steps(self):
        self.next_step_id = 0
        self.last_step_id = -1
        self.last_resp = None


class Err409(Exception):
    pass


class FakeEnv:
    def __init__(self):
        self.applied = 0          # how many times env.step actually ran

    def step(self, text):
        self.applied += 1
        return {"obs": f"obs#{self.applied}", "text": text}


async def handle_step(sess, step_id, text, work_delay=0.0):
    """Mirror of the server /step handler (lock + single-slot cache + 409)."""
    async with sess.lock:
        if step_id >= 0:
            if step_id == sess.last_step_id:
                return sess.last_resp
            if step_id != sess.next_step_id:
                raise Err409(f"step_id {step_id} != expected {sess.next_step_id}")
        # simulate `await asyncio.to_thread(env.step)` — yields the loop so a racing retry
        # can reach (and block on) the lock while we're "computing".
        await asyncio.sleep(work_delay)
        resp = sess.env.step(text)
        if step_id >= 0:
            sess.last_step_id = step_id
            sess.last_resp = resp
            sess.next_step_id = step_id + 1
        return resp


async def main():
    # 1. normal monotone sequence
    s = _Session(FakeEnv())
    for k in range(5):
        await handle_step(s, k, f"a{k}")
    assert s.env.applied == 5, s.env.applied
    assert s.next_step_id == 5 and s.last_step_id == 4
    print("1. monotone sequence: env applied 5/5 ... OK")

    # 2. replay of the latest id -> cached, no re-step
    r_first = s.last_resp
    r_replay = await handle_step(s, 4, "a4")
    assert s.env.applied == 5, "replay must NOT re-step"
    assert r_replay == r_first, "replay must return the cached response"
    print("2. replay latest id: cached, env not re-stepped ... OK")

    # 3. out-of-order id -> 409
    try:
        await handle_step(s, 99, "bogus")
        raise AssertionError("expected 409 for out-of-order id")
    except Err409:
        pass
    assert s.env.applied == 5, "409 must not apply the env"
    print("3. out-of-order id: 409, env untouched ... OK")

    # 4. concurrent duplicate (retry races the original) -> exactly once
    s2 = _Session(FakeEnv())
    # both carry step_id=0; the lock must serialize them so the env steps once
    res = await asyncio.gather(
        handle_step(s2, 0, "x", work_delay=0.05),
        handle_step(s2, 0, "x", work_delay=0.05),
    )
    assert s2.env.applied == 1, f"concurrent dup applied {s2.env.applied} (must be 1)"
    assert res[0] == res[1], "both racers must get the same response"
    assert s2.next_step_id == 1
    print("4. concurrent duplicate retry: env applied 1, identical resp ... OK")

    # 5. /reset restarts the counter
    s2.reset_steps()
    assert s2.next_step_id == 0 and s2.last_step_id == -1 and s2.last_resp is None
    await handle_step(s2, 0, "y")
    assert s2.env.applied == 2 and s2.next_step_id == 1
    print("5. reset_steps restarts counter ... OK")

    # 6. legacy id=-1 path applies every call, keeps no cache
    s3 = _Session(FakeEnv())
    await handle_step(s3, -1, "p")
    await handle_step(s3, -1, "p")
    assert s3.env.applied == 2, "legacy -1 must apply every call"
    assert s3.next_step_id == 0 and s3.last_step_id == -1, "legacy must not populate cache"
    print("6. legacy id=-1: applies every call, no cache ... OK")

    print("\nALL IDEMPOTENCY TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
