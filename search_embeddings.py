#!/usr/bin/env python3
"""
Search YOKATLAS programs using semantic embeddings with parameter-based queries.
Constructs query text exactly like the embedding creation process.
"""

import json
import os
import numpy as np
from typing import List, Dict, Any, Optional
import argparse

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(
        "⚠️  sentence-transformers not installed. Install with: pip install sentence-transformers"
    )

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  faiss not installed. Using numpy-based search (slower)")


class YOKATLASEmbeddingSearch:
    """Search YOKATLAS programs using semantic embeddings with parameters."""

    def __init__(self, embeddings_dir: str):
        """
        Initialize the search engine.

        Args:
            embeddings_dir: Directory containing embeddings and related files
        """
        self.embeddings_dir = embeddings_dir
        self.model = None
        self.embeddings = None
        self.programs = None
        self.texts = None
        self.metadata = None
        self.faiss_index = None

        self._load_data()

    def _load_data(self):
        """Load all necessary data from the embeddings directory."""
        print(f"📂 Loading embedding data from: {self.embeddings_dir}")

        # Load metadata
        metadata_file = os.path.join(self.embeddings_dir, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            print(f"📊 Loaded metadata: {self.metadata['total_programs']} programs")
        else:
            print("⚠️  metadata.json not found")

        # Load embeddings
        embeddings_file = os.path.join(self.embeddings_dir, "embeddings.npy")
        if os.path.exists(embeddings_file):
            self.embeddings = np.load(embeddings_file)
            print(f"🧮 Loaded embeddings: {self.embeddings.shape}")
        else:
            raise FileNotFoundError(
                f"embeddings.npy not found in {self.embeddings_dir}"
            )

        # Load programs
        programs_file = os.path.join(self.embeddings_dir, "programs.json")
        if os.path.exists(programs_file):
            with open(programs_file, "r", encoding="utf-8") as f:
                self.programs = json.load(f)
            print(f"📋 Loaded programs data: {len(self.programs)} programs")
        else:
            raise FileNotFoundError(f"programs.json not found in {self.embeddings_dir}")

        # Load texts
        texts_file = os.path.join(self.embeddings_dir, "texts.json")
        if os.path.exists(texts_file):
            with open(texts_file, "r", encoding="utf-8") as f:
                self.texts = json.load(f)
            print(f"📝 Loaded text representations: {len(self.texts)} texts")

        # Load SentenceTransformer model
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.metadata:
            model_name = self.metadata.get(
                "model_name", "paraphrase-multilingual-MiniLM-L12-v2"
            )
            print(f"🤖 Loading SentenceTransformer model: {model_name}")
            self.model = SentenceTransformer(model_name)

        # Load FAISS index if available
        faiss_file = os.path.join(self.embeddings_dir, "faiss_index.bin")
        if FAISS_AVAILABLE and os.path.exists(faiss_file):
            print("🔍 Loading FAISS index for fast search...")
            self.faiss_index = faiss.read_index(faiss_file)
            print(f"✅ FAISS index loaded: {self.faiss_index.ntotal} vectors")

        print("✅ All data loaded successfully!\n")

    def params_to_query_text(self, params: Dict[str, Any]) -> str:
        """
        Convert search parameters to query text using the same format as embedding creation.

        Args:
            params: Search parameters dictionary

        Returns:
            Query text in the same format used for embeddings
        """
        parts = []

        # Map parameter names to the format used in embeddings
        param_mapping = {
            "universite": "Üniversite",
            "uni_adi": "Üniversite",
            "program": "Program",
            "program_adi": "Program",
            "fakulte": "Fakülte",
            "program_detay": "Detay",
            "sehir": "Şehir",
            "sehir_adi": "Şehir",
            "universite_turu": "Tür",
            "ucret_burs": "Ücret",
            "ogretim_turu": "Öğretim",
        }

        # Add parameters in the same order as embedding creation
        for param_key, display_name in param_mapping.items():
            if param_key in params and params[param_key]:
                value = params[param_key]
                # Handle list values (convert to string)
                if isinstance(value, list):
                    if value:  # Only add if list is not empty
                        value = ", ".join(str(v) for v in value)
                    else:
                        continue
                parts.append(f"{display_name}: {value}")

        # Add ranking/score info if provided
        if "tbs" in params and params["tbs"]:
            parts.append(f"TBS: {params['tbs']}")

        if "taban" in params and params["taban"]:
            parts.append(f"Taban: {params['taban']}")

        query_text = " | ".join(parts)
        return query_text

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a search query into an embedding vector.

        Args:
            query: Search query text

        Returns:
            Query embedding vector
        """
        if not self.model:
            raise RuntimeError("SentenceTransformer model not available")

        return self.model.encode([query], convert_to_numpy=True)[0]

    def search_with_params(
        self,
        params: Dict[str, Any],
        k: int = 10,
        puan_turu_filter: Optional[str] = None,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search for programs using parameters.

        Args:
            params: Search parameters dictionary
            k: Number of results to return
            puan_turu_filter: Filter by specific puan_turu (say, ea, söz, dil)
            min_similarity: Minimum similarity score to include

        Returns:
            List of similar programs with similarity scores
        """
        # Convert params to query text
        query_text = self.params_to_query_text(params)
        print(f"🔍 Query: '{query_text}'")

        if puan_turu_filter:
            print(f"   Filtering by puan_turu: {puan_turu_filter}")

        # Encode the query
        query_embedding = self.encode_query(query_text)

        # Use FAISS if available
        if self.faiss_index:
            return self._search_with_faiss(
                query_embedding, k, puan_turu_filter, min_similarity
            )
        else:
            return self._search_with_numpy(
                query_embedding, k, puan_turu_filter, min_similarity
            )

    def _search_with_faiss(
        self,
        query_embedding: np.ndarray,
        k: int,
        puan_turu_filter: Optional[str],
        min_similarity: float,
    ) -> List[Dict[str, Any]]:
        """Search using FAISS index (faster)."""
        # Normalize query embedding for cosine similarity
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # If we need to filter, we might need more results initially
        search_k = k * 10 if puan_turu_filter else k
        search_k = min(search_k, self.faiss_index.ntotal)

        # Search
        similarities, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1).astype("float32"), search_k
        )

        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if similarity < min_similarity:
                continue

            program = self.programs[idx].copy()

            # Apply puan_turu filter
            if puan_turu_filter and program.get("puan_turu") != puan_turu_filter:
                continue

            program["similarity"] = float(similarity)
            program["search_text"] = self.texts[idx] if self.texts else ""
            results.append(program)

            if len(results) >= k:
                break

        return results

    def _search_with_numpy(
        self,
        query_embedding: np.ndarray,
        k: int,
        puan_turu_filter: Optional[str],
        min_similarity: float,
    ) -> List[Dict[str, Any]]:
        """Search using numpy (slower but always available)."""
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )
        query_normalized = query_embedding / np.linalg.norm(query_embedding)

        # Compute similarities
        similarities = np.dot(normalized_embeddings, query_normalized)

        # Get top k indices
        top_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in top_indices:
            similarity = similarities[idx]

            if similarity < min_similarity:
                continue

            program = self.programs[idx].copy()

            # Apply puan_turu filter
            if puan_turu_filter and program.get("puan_turu") != puan_turu_filter:
                continue

            program["similarity"] = float(similarity)
            program["search_text"] = self.texts[idx] if self.texts else ""
            results.append(program)

            if len(results) >= k:
                break

        return results

    def print_search_results(
        self, results: List[Dict[str, Any]], show_details: bool = False
    ):
        """
        Print search results in a formatted way.

        Args:
            results: List of search results
            show_details: Whether to show detailed information
        """
        if not results:
            print("❌ No results found.")
            return

        print(f"\n📊 Found {len(results)} results:")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            similarity = result.get("similarity", 0)
            uni_adi = result.get("uni_adi", "N/A")
            program_adi = result.get("program_adi", "N/A")
            sehir_adi = result.get("sehir_adi", "N/A")
            puan_turu = result.get("puan_turu", "N/A")

            print(f"{i:2d}. [{similarity:.3f}] {uni_adi}")
            print(f"     📚 {program_adi}")
            print(f"     📍 {sehir_adi} | 📊 {puan_turu.upper()}")

            if show_details:
                fakulte = result.get("fakulte", "N/A")
                program_detay = result.get("program_detay", "N/A")
                ucret_burs = result.get("ucret_burs", "N/A")

                # Get recent TBS and taban
                tbs_data = result.get("tbs", {})
                tbs = "N/A"
                for year in ["2025", "2024", "2023"]:
                    if year in tbs_data and tbs_data[year] and tbs_data[year] != "---":
                        tbs = tbs_data[year]
                        break

                taban_data = result.get("taban", {})
                taban = "N/A"
                for year in ["2025", "2024", "2023"]:
                    if (
                        year in taban_data
                        and taban_data[year]
                        and taban_data[year] != "---"
                    ):
                        taban = taban_data[year]
                        break

                print(f"     🏫 {fakulte}")
                print(f"     📝 {program_detay}")
                print(f"     💰 {ucret_burs} | 🎯 TBS: {tbs} | 📈 Taban: {taban}")

            print()


def main():
    parser = argparse.ArgumentParser(
        description="Search YOKATLAS programs using semantic embeddings with parameters"
    )
    parser.add_argument(
        "--embeddings", "-e", help="Embeddings directory (default: auto-detect latest)"
    )
    parser.add_argument("--universite", help="University name")
    parser.add_argument("--program", help="Program name")
    parser.add_argument("--sehir", help="City name")
    parser.add_argument("--fakulte", help="Faculty name")
    parser.add_argument("--universite-turu", help="University type (Devlet/Vakıf)")
    parser.add_argument("--ucret-burs", help="Fee/Scholarship status")
    parser.add_argument(
        "--k",
        "-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)",
    )
    parser.add_argument(
        "--puan-turu",
        "-p",
        choices=["say", "ea", "söz", "dil"],
        help="Filter by puan_turu",
    )
    parser.add_argument(
        "--min-similarity",
        "-s",
        type=float,
        default=0.0,
        help="Minimum similarity score (default: 0.0)",
    )
    parser.add_argument(
        "--details", "-d", action="store_true", help="Show detailed information"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive parameter entry mode",
    )

    args = parser.parse_args()

    # Find embeddings directory if not specified
    if not args.embeddings:
        embeddings_dirs = [d for d in os.listdir(".") if d.startswith("embeddings_")]
        if embeddings_dirs:
            args.embeddings = sorted(embeddings_dirs)[-1]
            print(f"🔍 Auto-detected embeddings: {args.embeddings}")
        else:
            print(
                "❌ No embeddings directory found. Please run create_embeddings.py first."
            )
            return

    # Initialize search engine
    try:
        search_engine = YOKATLASEmbeddingSearch(args.embeddings)
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return

    # Interactive mode
    if args.interactive:
        print("🎯 Interactive Parameter Search Mode")
        print("Commands:")
        print("  set <param> <value>  - Set a parameter (e.g., 'set universite Koç')")
        print("  remove <param>       - Remove a parameter")
        print("  show                 - Show current parameters and settings")
        print("  search               - Perform search with current parameters")
        print("  clear                - Clear all parameters")
        print("  k <number>           - Set number of results")
        print("  filter <puan_turu>   - Set puan_turu filter (say/ea/söz/dil/none)")
        print("  details              - Toggle detailed view")
        print("  help                 - Show this help")
        print("  quit                 - Exit")
        print()

        # Initialize interactive settings
        params = {}
        k = args.k
        puan_turu_filter = args.puan_turu
        show_details = args.details
        min_similarity = args.min_similarity

        # Add command line params if provided
        if args.universite:
            params["universite"] = args.universite
        if args.program:
            params["program"] = args.program
        if args.sehir:
            params["sehir"] = args.sehir
        if args.fakulte:
            params["fakulte"] = args.fakulte
        if args.universite_turu:
            params["universite_turu"] = args.universite_turu
        if args.ucret_burs:
            params["ucret_burs"] = args.ucret_burs

        def show_current_settings():
            print("\n📋 Current Settings:")
            print("=" * 40)
            print("Parameters:")
            if params:
                for key, value in params.items():
                    print(f"  {key}: {value}")
            else:
                print("  (no parameters set)")

            print(f"\nSearch Settings:")
            print(f"  k (results): {k}")
            print(f"  puan_turu filter: {puan_turu_filter or 'none'}")
            print(f"  min_similarity: {min_similarity}")
            print(f"  details: {'ON' if show_details else 'OFF'}")

            if params:
                query_preview = search_engine.params_to_query_text(params)
                print(f"\nQuery Preview: '{query_preview}'")
            print()

        # Show initial settings
        show_current_settings()

        while True:
            try:
                command = input("🔍 Enter command: ").strip()

                if not command:
                    continue

                parts = command.split(None, 2)
                cmd = parts[0].lower()

                if cmd == "quit" or cmd == "exit":
                    print("👋 Goodbye!")
                    break

                elif cmd == "help":
                    print("\n📖 Available Commands:")
                    print("  set universite <name>     - Set university name")
                    print("  set program <name>        - Set program name")
                    print("  set sehir <name>          - Set city name")
                    print("  set fakulte <name>        - Set faculty name")
                    print(
                        "  set universite_turu <type> - Set university type (Devlet/Vakıf)"
                    )
                    print("  set ucret_burs <status>   - Set fee/scholarship status")
                    print("  remove <param>            - Remove a parameter")
                    print("  show                      - Show current settings")
                    print(
                        "  search                    - Search with current parameters"
                    )
                    print("  clear                     - Clear all parameters")
                    print("  k <number>                - Set number of results")
                    print(
                        "  filter <puan_turu>        - Set filter (say/ea/söz/dil/none)"
                    )
                    print("  details                   - Toggle detailed view")
                    print("  quit                      - Exit")
                    print()

                elif cmd == "set":
                    if len(parts) < 3:
                        print("❌ Usage: set <parameter> <value>")
                        print(
                            "   Parameters: universite, program, sehir, fakulte, universite_turu, ucret_burs"
                        )
                        continue

                    param = parts[1].lower()
                    value = parts[2]

                    valid_params = [
                        "universite",
                        "program",
                        "sehir",
                        "fakulte",
                        "universite_turu",
                        "ucret_burs",
                    ]
                    if param in valid_params:
                        params[param] = value
                        print(f"✅ Set {param} = '{value}'")
                    else:
                        print(
                            f"❌ Invalid parameter '{param}'. Valid: {', '.join(valid_params)}"
                        )

                elif cmd == "remove":
                    if len(parts) < 2:
                        print("❌ Usage: remove <parameter>")
                        continue

                    param = parts[1].lower()
                    if param in params:
                        del params[param]
                        print(f"✅ Removed {param}")
                    else:
                        print(f"❌ Parameter '{param}' not set")

                elif cmd == "show":
                    show_current_settings()

                elif cmd == "clear":
                    params.clear()
                    print("✅ All parameters cleared")

                elif cmd == "search":
                    if not params:
                        print(
                            "❌ No parameters set. Use 'set <param> <value>' to add parameters."
                        )
                        continue

                    print("🔍 Searching...")
                    results = search_engine.search_with_params(
                        params, k, puan_turu_filter, min_similarity
                    )
                    search_engine.print_search_results(results, show_details)

                elif cmd == "k":
                    if len(parts) < 2:
                        print(f"Current k: {k}")
                        continue

                    try:
                        new_k = int(parts[1])
                        k = new_k
                        print(f"✅ Set k = {k}")
                    except ValueError:
                        print("❌ Invalid number")

                elif cmd == "filter":
                    if len(parts) < 2:
                        print(f"Current filter: {puan_turu_filter or 'none'}")
                        continue

                    new_filter = parts[1].lower()
                    if new_filter in ["say", "ea", "söz", "dil", "none"]:
                        puan_turu_filter = None if new_filter == "none" else new_filter
                        print(f"✅ Filter set to: {puan_turu_filter or 'none'}")
                    else:
                        print("❌ Invalid filter. Use: say, ea, söz, dil, or none")

                elif cmd == "details":
                    show_details = not show_details
                    print(f"✅ Details view: {'ON' if show_details else 'OFF'}")

                else:
                    print(
                        f"❌ Unknown command '{cmd}'. Type 'help' for available commands."
                    )

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

        return

    # Command line mode
    # Build search parameters from command line arguments
    params = {}
    if args.universite:
        params["universite"] = args.universite
    if args.program:
        params["program"] = args.program
    if args.sehir:
        params["sehir"] = args.sehir
    if args.fakulte:
        params["fakulte"] = args.fakulte
    if args.universite_turu:
        params["universite_turu"] = args.universite_turu
    if args.ucret_burs:
        params["ucret_burs"] = args.ucret_burs

    if not params:
        print(
            "❌ Please provide at least one search parameter or use --interactive mode."
        )
        print(
            "Example: python search_embeddings.py --universite 'Koç' --program 'Bilgisayar'"
        )
        print("Or: python search_embeddings.py --interactive")
        return

    # Perform search
    results = search_engine.search_with_params(
        params, args.k, args.puan_turu, args.min_similarity
    )
    search_engine.print_search_results(results, args.details)


if __name__ == "__main__":
    main()
