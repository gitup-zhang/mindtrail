import asyncio
import importlib.util
import os
import sys
import tempfile
from pathlib import Path


def load_local_package():
    package_root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "mindtrail",
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load local mindtrail package")

    module = importlib.util.module_from_spec(spec)
    sys.modules["mindtrail"] = module
    spec.loader.exec_module(module)
    return module


demo_db = Path(tempfile.gettempdir()) / "mindtrail-demo.db"
os.environ.setdefault("MINDTRAIL_DB_URL", f"sqlite:///{demo_db.as_posix()}")

for suffix in ("", "-wal", "-shm", "-journal"):
    path = demo_db.with_name(demo_db.name + suffix)
    if os.path.exists(path):
        os.remove(path)


MindTrail = load_local_package().MindTrail


async def main() -> None:
    engine = MindTrail(user="demo-user")
    await engine.delete_all()

    await engine.add(
        "MindTrail is a long-term memory engine that combines multi-sector vectors, waypoint recall, and temporal facts.",
        tags=["memory-engine"],
        meta={"source": "demo", "sector": "semantic"},
    )
    await engine.add(
        "The engine stores long documents as root-child memories so retrieval can keep both summary and detail.",
        tags=["architecture"],
        meta={"source": "demo", "sector": "semantic"},
    )
    await engine.add(
        "The engine uses salience reinforcement and waypoint-aware retrieval to make important memories easier to recall.",
        tags=["retrieval"],
        meta={"source": "demo", "sector": "semantic"},
    )

    await engine.remember_fact(
        subject="MindTrail",
        predicate="stage",
        object_value="showcase prototype",
        metadata={"source": "demo"},
    )

    results = await engine.search("What does this memory engine do?", limit=3)
    reflection = await engine.reflect(limit=20, min_cluster_size=2, similarity_threshold=0.14)
    decay = await engine.decay(limit=20)
    timeline = await engine.subject_timeline("MindTrail")

    print("Search results:")
    for item in results:
        print(f"- {item['score']:.3f} | {item['primary_sector']} | {item['content']}")

    print("\nReflection:")
    print(reflection)

    print("\nDecay:")
    print(decay)

    print("\nTimeline:")
    for event in timeline:
        print(f"- {event['change_type']} | {event['predicate']} -> {event['object']}")


if __name__ == "__main__":
    asyncio.run(main())
