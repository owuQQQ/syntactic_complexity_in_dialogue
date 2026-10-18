import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from syntactic_complexity_convergence.cka import analyze as analyze_cka
from syntactic_complexity_convergence.cka import linear_cka
from syntactic_complexity_convergence.calculate_metrics import (
    calculate_with_pipeline,
    calculate_complexity,
    calculate_syncom,
    dependency_length,
    dependency_metrics,
    long_dependency_ratio,
    sentence_syncom_metrics,
    summarize_syncom,
    tree_depth,
)
from syntactic_complexity_convergence.convergence import analyze as analyze_convergence
from syntactic_complexity_convergence.dialogue_lengths import calculate_lengths
from syntactic_complexity_convergence.fouriergpt import (
    _length_match,
    analyze as analyze_detection,
    load_nll_series,
    load_semcom_surprisal,
    load_surprisal_series,
    series_to_spectra,
)
from syntactic_complexity_convergence.phase_labels import add_phase_labels
from syntactic_complexity_convergence.preprocess import preprocess
from syntactic_complexity_convergence.statistics import benjamini_hochberg, phase_permutation_test


class PipelineTests(unittest.TestCase):
    def test_preprocess_csv_maps_speakers_and_orders_turns(self) -> None:
        """Normalize CSV headers, order turns, and map speaker roles."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.csv"
            output = root / "turns.csv"
            pd.DataFrame([
                {"id": "d1", "speaker": "B", "turn_id": 2, "utterance": "second"},
                {"id": "d1", "speaker": "A", "turn_id": 1, "utterance": "first"},
            ]).to_csv(source, index=False)
            result = preprocess(source, output)
            self.assertEqual(result["speaker"].tolist(), ["initiator", "follower"])
            self.assertEqual(result["text"].tolist(), ["first", "second"])

    def test_preprocess_role_specific_legacy_json(self) -> None:
        """Keep a legacy role-specific JSON input assigned to its declared role."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "legacy.json"
            output = root / "turns.csv"
            source.write_text(json.dumps({"dialogue": ["one", "two"]}), encoding="utf-8")
            result = preprocess(source, output, role="follower")
            self.assertEqual(result["speaker"].unique().tolist(), ["follower"])

    def test_dialogue_lengths_aggregate_word_and_turn_counts(self) -> None:
        """Aggregate turn-level token counts into dialogue lengths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "metrics.csv"
            output = root / "lengths.csv"
            pd.DataFrame([
                {"dialogue_id": "d1", "word_count": 5},
                {"dialogue_id": "d1", "word_count": 7},
                {"dialogue_id": "d2", "word_count": 3},
            ]).to_csv(source, index=False)
            result = calculate_lengths(source, output).set_index("dialogue_id")
            self.assertEqual(result.loc["d1", "word_count"], 12)
            self.assertEqual(result.loc["d1", "turn_count"], 2)

    def test_phase_labels_assign_eight_and_sixteen_ordered_bins(self) -> None:
        """Assign balanced, chronological 8- and 16-phase labels."""
        frame = pd.DataFrame({
            "dialogue_id": ["d1"] * 32,
            "speaker": ["initiator"] * 32,
            "turn_index": list(reversed(range(32))),
        })
        result = add_phase_labels(frame)
        self.assertEqual(result.groupby("phase_8").size().tolist(), [4] * 8)
        self.assertEqual(result.groupby("phase_16").size().tolist(), [2] * 16)
        ordered = result.sort_values("turn_index")
        self.assertTrue(ordered["phase_8"].is_monotonic_increasing)

    def test_linear_cka_is_one_for_same_nonconstant_data(self) -> None:
        """Verify identical feature representations have unit linear CKA."""
        x = np.asarray([[1, 2], [2, 1], [3, 5], [4, 2]], dtype=float)
        self.assertAlmostEqual(linear_cka(x, x), 1.0)

    def test_dependency_metrics_calculation(self) -> None:
        """Check the original tree, length, complexity, and long-arc formulas."""
        words = [
            SimpleNamespace(id=1, head=0),
            SimpleNamespace(id=2, head=1),
            SimpleNamespace(id=3, head=1),
        ]

        class Pipeline:
            """Return the same three-token dependency tree for each input."""

            def __call__(self, text: str) -> SimpleNamespace:
                """Return a parser-shaped document for the metric calculation."""
                return SimpleNamespace(sentences=[SimpleNamespace(words=words)])

        pipeline = Pipeline()
        metrics = dependency_metrics("a b c", pipeline)
        self.assertEqual(metrics["word_count"], 3)
        self.assertEqual(metrics["tree_depth"], 2)
        self.assertAlmostEqual(metrics["syn_value"], 1.75)
        self.assertAlmostEqual(metrics["dependency_length"], 1.5)
        self.assertEqual(tree_depth("a b c", pipeline), 2)
        self.assertAlmostEqual(dependency_length("a b c", pipeline), 1.5)
        self.assertAlmostEqual(calculate_complexity("a b c", pipeline), 1.75)
        self.assertAlmostEqual(long_dependency_ratio("a b c", pipeline), 0.0)
        np.testing.assert_allclose(
            sentence_syncom_metrics("a b c", pipeline),
            [1.75, 1.5, 2.0, 0.0],
        )
        np.testing.assert_allclose(
            calculate_syncom(["a b c", ""], pipeline),
            [[1.75, 1.5, 2.0, 0.0]],
        )
        np.testing.assert_allclose(
            summarize_syncom(["a b c"], pipeline),
            [1.75, 1.5, 2.0, 0.0],
        )

    def test_stanza_pipeline_writer_saves_metrics_per_turn(self) -> None:
        """Apply an injected Stanza-shaped pipeline and write all computed metrics."""
        words = [
            SimpleNamespace(id=1, head=0),
            SimpleNamespace(id=2, head=1),
            SimpleNamespace(id=3, head=1),
        ]

        class Pipeline:
            def __call__(self, text: str) -> SimpleNamespace:
                """Return a minimal dependency parse for one utterance."""
                return SimpleNamespace(sentences=[SimpleNamespace(words=words)])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "turns.csv"
            output_path = root / "metrics.csv"
            pd.DataFrame([{
                "dialogue_id": "d1",
                "speaker": "initiator",
                "turn_index": 0,
                "text": "a b c",
            }]).to_csv(input_path, index=False)
            result = calculate_with_pipeline(input_path, output_path, Pipeline())
            self.assertTrue(output_path.exists())
            self.assertAlmostEqual(result.loc[0, "syn_value"], 1.75)

    def test_phase_permutation_detects_ordered_decline_reproducibly(self) -> None:
        """Detect the seeded negative trend and reproduce its permutation p-value."""
        gaps = np.asarray([
            [8, 6, 4, 2],
            [9, 7, 5, 3],
            [7, 5, 3, 1],
            [10, 8, 6, 4],
        ], dtype=float)
        first = phase_permutation_test(gaps, permutations=999, seed=7)
        second = phase_permutation_test(gaps, permutations=999, seed=7)
        self.assertLess(first[0], 0)
        self.assertLess(first[1], 0.05)
        self.assertEqual(first[1], second[1])

    def test_convergence_writes_phase_and_test_outputs(self) -> None:
        """Write phase gaps and permutation-test results from paired dialogues."""
        rows = []
        for dialogue in range(4):
            for role, offset in (("initiator", 1.0), ("follower", 9.0 - dialogue * 0.1)):
                for turn in range(8):
                    rows.append({
                        "dialogue_id": str(dialogue),
                        "speaker": role,
                        "turn_index": turn,
                        "metric": offset - turn * 0.7 if role == "follower" else offset + turn * 0.1,
                    })
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "metrics.csv"
            prefix = root / "analysis"
            pd.DataFrame(rows).to_csv(input_path, index=False)
            _, summary = analyze_convergence(
                input_path,
                prefix,
                metrics=["metric"],
                n_phases=4,
                min_turns=8,
                permutations=99,
            )
            self.assertTrue(prefix.with_name("analysis_phases.csv").exists())
            self.assertTrue(prefix.with_name("analysis_tests.csv").exists())
            self.assertIn("permutation_p_value_bh", summary)

    def test_cka_writes_phase_distances_and_permutation_test(self) -> None:
        """Write CKA distances and a test of their chronological trend."""
        rows = []
        for dialogue in range(3):
            for role, offset in (("initiator", 0.0), ("follower", 0.8 + dialogue * 0.1)):
                for turn in range(16):
                    rows.append({
                        "dialogue_id": str(dialogue),
                        "speaker": role,
                        "turn_index": turn,
                        "syn_value": np.sin(turn + offset),
                        "dependency_length": np.cos(turn / 2 + offset),
                        "tree_depth": turn % 4 + offset,
                        "complex_dependency_ratio": (turn % 5) / 5 + offset,
                    })
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "metrics.csv"
            prefix = root / "analysis"
            pd.DataFrame(rows).to_csv(input_path, index=False)
            phases, summary = analyze_cka(
                input_path,
                prefix,
                n_phases=4,
                min_turns=8,
                points_per_phase=4,
                permutations=99,
            )
            self.assertEqual(len(phases), 4)
            self.assertTrue(prefix.with_name("analysis_cka_test.csv").exists())
            self.assertIn("permutation_p_value", summary)

    def test_length_matching_balances_each_bin(self) -> None:
        """Retain equal class counts within the selected length bins."""
        records = [
            (f"h{i}", 0, float(i), np.asarray([1.0, 2.0]))
            for i in range(8)
        ] + [
            (f"l{i}", 1, float(i), np.asarray([3.0, 4.0]))
            for i in range(4, 12)
        ]
        matched = _length_match(records, bins=2, seed=1)
        self.assertEqual(
            sum(record[1] == 0 for record in matched),
            sum(record[1] == 1 for record in matched),
        )

    def test_surprisal_series_reader_rejects_bad_rows_and_reads_tab_series(self) -> None:
        """Load tab-separated dialogue-level surprisal values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "series.txt"
            path.write_text("dialogue-1\t0.2 0.4 0.6\n", encoding="utf-8")
            values = load_surprisal_series(path)
            np.testing.assert_allclose(values["dialogue-1"], [0.2, 0.4, 0.6])

    def test_nll_series_matches_positive_half_fft_magnitudes(self) -> None:
        """Calculate one-sided Fourier magnitudes from dialogue NLL sequences."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nll.txt"
            path.write_text("dialogue-1\t1 2 3 4\n", encoding="utf-8")
            values = load_nll_series(path)
            spectrum = series_to_spectra(values)["dialogue-1"]
            np.testing.assert_allclose(
                spectrum,
                np.abs(np.fft.fft([1, 2, 3, 4])[:3]),
            )

    def test_semcom_reader_extracts_surprisal_and_ignores_other_metrics(self) -> None:
        """Extract dialogue surprisal from semcom output and ignore other metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "semcom.txt"
            path.write_text(
                "index:d1\nsurprisal\n1.0 2.0\nshannon\n4.0 5.0\n"
                "surprisal\n3.0\nindex:d2\nsurprisal\n6.0 7.0\n",
                encoding="utf-8",
            )
            series = load_semcom_surprisal(path)
            np.testing.assert_allclose(series["d1"], [1.0, 2.0, 3.0])
            np.testing.assert_allclose(series["d2"], [6.0, 7.0])

    def test_fourier_nll_surprisal_and_length_controls(self) -> None:
        """Run Fourier, NLL-derived Fourier, and surprisal detection controls."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            human_fft = []
            llm_fft = []
            human_lengths = []
            llm_lengths = []
            human_semcom = []
            llm_semcom = []
            human_nll = []
            llm_nll = []
            for index in range(20):
                length = 10 + index
                human_id = f"h{index}"
                llm_id = f"m{index}"
                human_lengths.append({"dialogue_id": human_id, "length": length})
                llm_lengths.append({"dialogue_id": llm_id, "length": length})
                human_semcom.append(
                    f"index:{human_id}\nsurprisal\n" + " ".join(["0.1"] * 12)
                    + "\nshannon\n0.5 0.5\n"
                )
                llm_semcom.append(
                    f"index:{llm_id}\nsurprisal\n" + " ".join(["2.0"] * 12)
                    + "\nshannon\n0.5 0.5\n"
                )
                human_nll.append(f"{human_id}\t" + " ".join(["0.1"] * 12))
                llm_nll.append(f"{llm_id}\t" + " ".join(["2.0"] * 12))
                for frequency in range(8):
                    human_fft.append({
                        "sid": human_id,
                        "freq": frequency,
                        "power": float(frequency == 1),
                    })
                    llm_fft.append({
                        "sid": llm_id,
                        "freq": frequency,
                        "power": float(frequency == 6),
                    })
            paths = {
                "human_fft": root / "human_fft.csv",
                "llm_fft": root / "llm_fft.csv",
                "human_lengths": root / "human_lengths.csv",
                "llm_lengths": root / "llm_lengths.csv",
                "human_semcom": root / "human_semcom.txt",
                "llm_semcom": root / "llm_semcom.txt",
                "human_nll": root / "human_nll.txt",
                "llm_nll": root / "llm_nll.txt",
            }
            pd.DataFrame(human_fft).to_csv(paths["human_fft"], index=False)
            pd.DataFrame(llm_fft).to_csv(paths["llm_fft"], index=False)
            pd.DataFrame(human_lengths).to_csv(paths["human_lengths"], index=False)
            pd.DataFrame(llm_lengths).to_csv(paths["llm_lengths"], index=False)
            paths["human_semcom"].write_text("\n".join(human_semcom), encoding="utf-8")
            paths["llm_semcom"].write_text("\n".join(llm_semcom), encoding="utf-8")
            paths["human_nll"].write_text("\n".join(human_nll), encoding="utf-8")
            paths["llm_nll"].write_text("\n".join(llm_nll), encoding="utf-8")
            result = analyze_detection(
                root / "scores.csv",
                human_fft=paths["human_fft"],
                llm_fft=paths["llm_fft"],
                human_lengths=paths["human_lengths"],
                llm_lengths=paths["llm_lengths"],
                human_semcom=paths["human_semcom"],
                llm_semcom=paths["llm_semcom"],
                width=16,
                length_bins=4,
            )
            nll_result = analyze_detection(
                root / "nll_scores.csv",
                human_lengths=paths["human_lengths"],
                llm_lengths=paths["llm_lengths"],
                human_nll=paths["human_nll"],
                llm_nll=paths["llm_nll"],
                width=16,
                length_bins=4,
            )
            self.assertIn("fouriergpt", result["model"].tolist())
            self.assertIn("surprisal_baseline", result["model"].tolist())
            self.assertIn("fouriergpt_nll", nll_result["model"].tolist())
            self.assertEqual(len(result), 4)
            self.assertEqual(len(nll_result), 2)

    def test_benjamini_hochberg_is_monotone_in_rank(self) -> None:
        """Ensure adjusted values respect sorted p-value ordering."""
        corrected = benjamini_hochberg([0.01, 0.04, 0.03])
        self.assertEqual(corrected, [0.03, 0.04, 0.04])


if __name__ == "__main__":
    unittest.main()
