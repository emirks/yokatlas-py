#!/usr/bin/env python3
"""
Complete example demonstrating YOKATLAS embedding workflow:
1. Search all programs (if not already done)
2. Create embeddings
3. Perform semantic search

This is a complete end-to-end demonstration.
"""

import os
import json
from datetime import datetime


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        import sentence_transformers
    except ImportError:
        missing.append("sentence-transformers")

    try:
        import faiss
    except ImportError:
        missing.append("faiss-cpu")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    if missing:
        print("❌ Missing dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        print("Or use: pip install -r requirements_embeddings.txt")
        return False

    return True


def find_latest_results_dir():
    """Find the latest results directory."""
    results_base = "yokatlas-results"
    if os.path.exists(results_base):
        subdirs = [
            d
            for d in os.listdir(results_base)
            if os.path.isdir(os.path.join(results_base, d))
            and d.startswith("all_programs_")
        ]
        if subdirs:
            return os.path.join(results_base, sorted(subdirs)[-1])
    return None


def find_latest_embeddings_dir():
    """Find the latest embeddings directory."""
    embeddings_dirs = [d for d in os.listdir(".") if d.startswith("embeddings_")]
    if embeddings_dirs:
        return sorted(embeddings_dirs)[-1]
    return None


def run_data_collection():
    """Run data collection if no results found."""
    print("🚀 Running data collection...")
    os.system("python simple_test.py")


def run_embedding_creation(results_dir):
    """Run embedding creation."""
    print(f"🧮 Creating embeddings from: {results_dir}")
    cmd = f'python create_embeddings.py --input "{results_dir}"'
    os.system(cmd)


def run_example_searches(embeddings_dir):
    """Run example searches to demonstrate functionality."""
    print(f"🔍 Running example searches with: {embeddings_dir}")

    # Import the search engine
    from search_embeddings import YOKATLASEmbeddingSearch

    try:
        search_engine = YOKATLASEmbeddingSearch(embeddings_dir)

        print("\n" + "=" * 60)
        print("🎯 EXAMPLE SEARCHES")
        print("=" * 60)

        # Example 1: Search for computer science programs
        print("\n1️⃣ Search: 'bilgisayar mühendisliği'")
        results = search_engine.search_similar_programs("bilgisayar mühendisliği", k=5)
        search_engine.print_search_results(results)

        # Example 2: Search for medical programs
        print("\n2️⃣ Search: 'tıp fakültesi'")
        results = search_engine.search_similar_programs("tıp fakültesi", k=5)
        search_engine.print_search_results(results)

        # Example 3: Search for engineering with filter
        print("\n3️⃣ Search: 'mühendislik' (SAY only)")
        results = search_engine.search_similar_programs(
            "mühendislik", k=5, puan_turu_filter="say"
        )
        search_engine.print_search_results(results)

        # Example 4: Search by university
        print("\n4️⃣ Search: 'boğaziçi üniversitesi'")
        results = search_engine.search_similar_programs("boğaziçi üniversitesi", k=5)
        search_engine.print_search_results(results)

        # Example 5: Search for economics/business
        print("\n5️⃣ Search: 'ekonomi işletme'")
        results = search_engine.search_similar_programs("ekonomi işletme", k=5)
        search_engine.print_search_results(results)

        print("\n" + "=" * 60)
        print("✨ Example searches completed!")
        print("=" * 60)

        # Show statistics
        if search_engine.metadata:
            print(f"\n📊 Dataset Statistics:")
            print(f"   Total programs: {search_engine.metadata['total_programs']:,}")
            print(
                f"   Embedding dimension: {search_engine.metadata['embedding_dimension']}"
            )
            print(f"   Model: {search_engine.metadata['model_name']}")

            puan_counts = search_engine.metadata.get("puan_turu_counts", {})
            print(f"\n   Programs by Puan Türü:")
            for puan_turu, count in puan_counts.items():
                print(f"     {puan_turu.upper()}: {count:,} programs")

        print(f"\n🎯 To run interactive search:")
        print(f"   python search_embeddings.py --interactive")
        print(f"\n🎯 To run single queries:")
        print(f"   python search_embeddings.py --query 'your search here' --k 10")

    except Exception as e:
        print(f"❌ Error running searches: {e}")


def main():
    print("🎉 YOKATLAS Embedding Workflow Example")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        return

    print("✅ All dependencies available")

    # Step 1: Check for existing data
    results_dir = find_latest_results_dir()

    if not results_dir:
        print("\n📥 No existing data found. Running data collection...")
        run_data_collection()
        results_dir = find_latest_results_dir()

        if not results_dir:
            print("❌ Data collection failed. Please run simple_test.py manually.")
            return

    print(f"✅ Using data from: {results_dir}")

    # Step 2: Check for existing embeddings
    embeddings_dir = find_latest_embeddings_dir()

    if not embeddings_dir:
        print("\n🧮 No existing embeddings found. Creating embeddings...")
        run_embedding_creation(results_dir)
        embeddings_dir = find_latest_embeddings_dir()

        if not embeddings_dir:
            print(
                "❌ Embedding creation failed. Please run create_embeddings.py manually."
            )
            return

    print(f"✅ Using embeddings from: {embeddings_dir}")

    # Step 3: Run example searches
    print("\n🔍 Running example searches...")
    run_example_searches(embeddings_dir)

    print(f"\n🎉 Complete workflow finished successfully!")
    print(f"   Data: {results_dir}")
    print(f"   Embeddings: {embeddings_dir}")


if __name__ == "__main__":
    main()
