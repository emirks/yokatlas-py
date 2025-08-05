#!/usr/bin/env python3
"""
Create embeddings from YOKATLAS program data and enable semantic search.
This script processes the JSON files from search results and creates vector embeddings.
"""

import json
import os
import numpy as np
import pickle
from typing import List, Dict, Any, Tuple
from datetime import datetime
import argparse
from pathlib import Path

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
    print("⚠️  faiss not installed. Install with: pip install faiss-cpu")


class YOKATLASEmbeddingCreator:
    """Create and manage embeddings for YOKATLAS program data."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize the embedding creator.

        Args:
            model_name: Name of the sentence transformer model to use
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )

        print(f"🤖 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.programs = None
        self.metadata = None

    def extract_program_text(self, program: Dict[str, Any]) -> str:
        """
        Extract meaningful text from a program entry for embedding.

        Args:
            program: Program dictionary from JSON

        Returns:
            Combined text representation of the program
        """
        parts = []

        # Core information
        if program.get("uni_adi"):
            parts.append(f"Üniversite: {program['uni_adi']}")

        if program.get("program_adi"):
            parts.append(f"Program: {program['program_adi']}")

        if program.get("fakulte"):
            parts.append(f"Fakülte: {program['fakulte']}")

        if program.get("program_detay"):
            parts.append(f"Detay: {program['program_detay']}")

        if program.get("sehir_adi"):
            parts.append(f"Şehir: {program['sehir_adi']}")

        if program.get("universite_turu"):
            parts.append(f"Tür: {program['universite_turu']}")

        if program.get("ucret_burs"):
            parts.append(f"Ücret: {program['ucret_burs']}")

        if program.get("ogretim_turu"):
            parts.append(f"Öğretim: {program['ogretim_turu']}")

        # Add recent ranking info if available
        tbs_data = program.get("tbs", {})
        for year in ["2025", "2024", "2023"]:
            if year in tbs_data and tbs_data[year] and tbs_data[year] != "---":
                parts.append(f"TBS {year}: {tbs_data[year]}")
                break

        # Add recent score info if available
        taban_data = program.get("taban", {})
        for year in ["2025", "2024", "2023"]:
            if year in taban_data and taban_data[year] and taban_data[year] != "---":
                parts.append(f"Taban {year}: {taban_data[year]}")
                break

        return " | ".join(parts)

    def load_programs_from_directory(self, results_dir: str) -> List[Dict[str, Any]]:
        """
        Load all programs from JSON files in a results directory.

        Args:
            results_dir: Path to directory containing JSON files

        Returns:
            List of all programs with puan_turu information
        """
        all_programs = []

        # Find all JSON files with program data
        json_files = [
            ("say", "lisans_programs_say.json"),
            ("ea", "lisans_programs_ea.json"),
            ("söz", "lisans_programs_söz.json"),
            ("dil", "lisans_programs_dil.json"),
        ]

        for puan_turu, filename in json_files:
            filepath = os.path.join(results_dir, filename)

            if os.path.exists(filepath):
                print(f"📂 Loading {filename}...")

                with open(filepath, "r", encoding="utf-8") as f:
                    programs = json.load(f)

                # Add puan_turu to each program
                for program in programs:
                    program["puan_turu"] = puan_turu
                    all_programs.append(program)

                print(f"   ✅ Loaded {len(programs)} programs from {puan_turu}")
            else:
                print(f"   ⚠️  File not found: {filepath}")

        print(f"\n📊 Total programs loaded: {len(all_programs)}")
        return all_programs

    def create_embeddings(
        self, programs: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create embeddings for all programs.

        Args:
            programs: List of program dictionaries

        Returns:
            Tuple of (embeddings array, text list)
        """
        print(f"\n🧮 Creating embeddings for {len(programs)} programs...")

        # Extract text for each program
        texts = []
        for i, program in enumerate(programs):
            text = self.extract_program_text(program)
            texts.append(text)

            if (i + 1) % 1000 == 0:
                print(f"   Processed {i + 1}/{len(programs)} programs...")

        print("🚀 Generating embeddings with SentenceTransformer...")

        # Create embeddings in batches for memory efficiency
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True)
            all_embeddings.append(batch_embeddings)

            print(
                f"   Embedded batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}"
            )

        # Combine all embeddings
        embeddings = np.vstack(all_embeddings)

        print(f"✅ Created embeddings: {embeddings.shape}")
        return embeddings, texts

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        programs: List[Dict[str, Any]],
        texts: List[str],
        output_dir: str,
    ):
        """
        Save embeddings and associated data.

        Args:
            embeddings: Numpy array of embeddings
            programs: List of program dictionaries
            texts: List of text representations
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save embeddings
        embeddings_file = os.path.join(output_dir, "embeddings.npy")
        np.save(embeddings_file, embeddings)
        print(f"💾 Saved embeddings to: {embeddings_file}")

        # Save programs data
        programs_file = os.path.join(output_dir, "programs.json")
        with open(programs_file, "w", encoding="utf-8") as f:
            json.dump(programs, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved programs to: {programs_file}")

        # Save texts
        texts_file = os.path.join(output_dir, "texts.json")
        with open(texts_file, "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved texts to: {texts_file}")

        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "total_programs": len(programs),
            "embedding_dimension": embeddings.shape[1],
            "model_name": self.model._modules["0"].auto_model.name_or_path,
            "puan_turu_counts": {},
        }

        # Count programs by puan_turu
        for program in programs:
            puan_turu = program.get("puan_turu", "unknown")
            metadata["puan_turu_counts"][puan_turu] = (
                metadata["puan_turu_counts"].get(puan_turu, 0) + 1
            )

        metadata_file = os.path.join(output_dir, "metadata.json")
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved metadata to: {metadata_file}")

        # Save FAISS index if available
        if FAISS_AVAILABLE:
            print("🔍 Creating FAISS index for fast similarity search...")

            # Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(
                dimension
            )  # Inner product (cosine similarity for normalized vectors)

            # Normalize embeddings for cosine similarity
            normalized_embeddings = embeddings / np.linalg.norm(
                embeddings, axis=1, keepdims=True
            )
            index.add(normalized_embeddings.astype("float32"))

            # Save FAISS index
            index_file = os.path.join(output_dir, "faiss_index.bin")
            faiss.write_index(index, index_file)
            print(f"💾 Saved FAISS index to: {index_file}")

        print(f"\n✨ All embedding data saved to: {os.path.abspath(output_dir)}")


def main():
    parser = argparse.ArgumentParser(
        description="Create embeddings from YOKATLAS program data"
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Input directory containing JSON files (e.g., yokatlas-results/all_programs_20241220_143022)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory for embeddings (default: embeddings_TIMESTAMP)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name",
    )

    args = parser.parse_args()

    # Find input directory if not specified
    if not args.input:
        results_base = "yokatlas-results"
        if os.path.exists(results_base):
            # Find the most recent results directory
            subdirs = [
                d
                for d in os.listdir(results_base)
                if os.path.isdir(os.path.join(results_base, d))
                and d.startswith("all_programs_")
            ]
            if subdirs:
                latest_dir = sorted(subdirs)[-1]
                args.input = os.path.join(results_base, latest_dir)
                print(f"🔍 Auto-detected input directory: {args.input}")
            else:
                print(
                    "❌ No results directories found. Please run simple_test.py first."
                )
                return
        else:
            print(
                "❌ No yokatlas-results directory found. Please run simple_test.py first."
            )
            return

    # Set output directory
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"embeddings_{timestamp}"

    print(f"🎯 Input: {args.input}")
    print(f"🎯 Output: {args.output}")
    print(f"🎯 Model: {args.model}")
    print()

    # Create embeddings
    creator = YOKATLASEmbeddingCreator(args.model)

    # Load programs
    programs = creator.load_programs_from_directory(args.input)

    if not programs:
        print("❌ No programs found. Please check the input directory.")
        return

    # Create embeddings
    embeddings, texts = creator.create_embeddings(programs)

    # Save everything
    creator.save_embeddings(embeddings, programs, texts, args.output)

    print(f"\n🎉 Embedding creation completed successfully!")
    print(f"   Total programs: {len(programs)}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    print(f"   Files saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
