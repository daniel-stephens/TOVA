import pytest
import math

from corpus_agent.config import CorpusSize
from corpus_agent.sizing import resolve_corpus_size, ResolvedCorpusSize


class TestResolveCorpusSize:
    def test_manual_mode_keeps_pydantic_defaults(self):
        cs = CorpusSize(num_instances=5)
        resolved = resolve_corpus_size(cs, num_plot_points=7, blur_level="medium", mode="manual")
        assert resolved.num_instances == 5
        assert resolved.waves_per_instance == 10
        assert resolved.emails_per_wave == 4
        assert resolved.emails_per_instance == 100

    def test_auto_mode_derives_from_plot_points_medium_blur(self):
        cs = CorpusSize(num_instances=10)
        resolved = resolve_corpus_size(cs, num_plot_points=7, blur_level="medium", mode="auto")
        assert resolved.num_instances == 10
        assert resolved.waves_per_instance == 7
        assert resolved.emails_per_wave == 4
        assert resolved.emails_per_instance == int(7 * 4 * 1.5)  # 42

    def test_auto_mode_light_blur(self):
        cs = CorpusSize(num_instances=5)
        resolved = resolve_corpus_size(cs, num_plot_points=8, blur_level="light", mode="auto")
        assert resolved.emails_per_wave == 3
        assert resolved.waves_per_instance == 8
        assert resolved.emails_per_instance == int(8 * 3 * 1.5)  # 36

    def test_auto_mode_heavy_blur(self):
        cs = CorpusSize(num_instances=5)
        resolved = resolve_corpus_size(cs, num_plot_points=6, blur_level="heavy", mode="auto")
        assert resolved.emails_per_wave == 5
        assert resolved.waves_per_instance == 6
        assert resolved.emails_per_instance == int(6 * 5 * 1.5)  # 45

    def test_explicit_values_override_auto(self):
        cs = CorpusSize(num_instances=10, waves_per_instance=3, emails_per_wave=2, emails_per_instance=20)
        resolved = resolve_corpus_size(cs, num_plot_points=8, blur_level="medium", mode="auto")
        assert resolved.waves_per_instance == 3
        assert resolved.emails_per_wave == 2
        assert resolved.emails_per_instance == 20

    def test_partial_explicit_in_auto_mode(self):
        cs = CorpusSize(num_instances=10, waves_per_instance=3)
        resolved = resolve_corpus_size(cs, num_plot_points=8, blur_level="medium", mode="auto")
        assert resolved.waves_per_instance == 3  # explicit wins
        assert resolved.emails_per_wave == 4  # auto-derived from blur
        assert resolved.emails_per_instance == int(3 * 4 * 1.5)  # derived from resolved waves * epw

    def test_explicit_tight_cap_respected(self):
        cs = CorpusSize(num_instances=10, emails_per_instance=10)
        resolved = resolve_corpus_size(cs, num_plot_points=8, blur_level="medium", mode="auto")
        assert resolved.emails_per_instance == 10  # user said 10, gets 10
        assert resolved.waves_per_instance == 8  # auto
        assert resolved.emails_per_wave == 4  # auto

    def test_returns_resolved_type(self):
        cs = CorpusSize(num_instances=1)
        resolved = resolve_corpus_size(cs, num_plot_points=5, blur_level="light", mode="auto")
        assert isinstance(resolved, ResolvedCorpusSize)

    def test_manual_mode_with_explicit_values(self):
        cs = CorpusSize(num_instances=5, waves_per_instance=3, emails_per_wave=2, emails_per_instance=50)
        resolved = resolve_corpus_size(cs, num_plot_points=99, blur_level="heavy", mode="manual")
        assert resolved.waves_per_instance == 3
        assert resolved.emails_per_wave == 2
        assert resolved.emails_per_instance == 50

    def test_auto_mode_unknown_blur_level_raises(self):
        cs = CorpusSize(num_instances=5)
        with pytest.raises(KeyError):
            resolve_corpus_size(cs, num_plot_points=7, blur_level="extreme", mode="auto")


class TestTotalEmailsSizing:
    def test_auto_total_emails_derives_num_instances(self):
        cs = CorpusSize(total_emails=500)
        resolved = resolve_corpus_size(cs, num_plot_points=7, blur_level="medium", mode="auto")
        # auto: 7 waves * 4 epw * 1.5 cap = 42 emails_per_instance
        # num_instances = ceil(500 / 42) = 12
        assert resolved.emails_per_instance == 42
        assert resolved.num_instances == math.ceil(500 / 42)

    def test_auto_total_emails_light_blur(self):
        cs = CorpusSize(total_emails=200)
        resolved = resolve_corpus_size(cs, num_plot_points=5, blur_level="light", mode="auto")
        # auto: 5 waves * 3 epw * 1.5 = 22 emails_per_instance (22.5 truncated)
        epi = int(5 * 3 * 1.5)  # 22
        assert resolved.emails_per_instance == epi
        assert resolved.num_instances == math.ceil(200 / epi)

    def test_auto_total_emails_heavy_blur(self):
        cs = CorpusSize(total_emails=1000)
        resolved = resolve_corpus_size(cs, num_plot_points=10, blur_level="heavy", mode="auto")
        # 10 waves * 5 epw * 1.5 = 75 emails_per_instance
        epi = int(10 * 5 * 1.5)  # 75
        assert resolved.emails_per_instance == epi
        assert resolved.num_instances == math.ceil(1000 / epi)

    def test_auto_total_emails_with_explicit_epi_override(self):
        cs = CorpusSize(total_emails=300, emails_per_instance=30)
        resolved = resolve_corpus_size(cs, num_plot_points=6, blur_level="medium", mode="auto")
        assert resolved.emails_per_instance == 30
        assert resolved.num_instances == math.ceil(300 / 30)

    def test_auto_total_emails_with_explicit_num_instances_wins(self):
        cs = CorpusSize(total_emails=500, num_instances=3)
        resolved = resolve_corpus_size(cs, num_plot_points=7, blur_level="medium", mode="auto")
        # Explicit num_instances overrides total_emails derivation
        assert resolved.num_instances == 3

    def test_manual_mode_ignores_total_emails(self):
        cs = CorpusSize(total_emails=999, num_instances=5)
        resolved = resolve_corpus_size(cs, num_plot_points=7, blur_level="medium", mode="manual")
        assert resolved.num_instances == 5

    def test_auto_no_total_emails_uses_explicit_num_instances(self):
        cs = CorpusSize(num_instances=10)
        resolved = resolve_corpus_size(cs, num_plot_points=7, blur_level="medium", mode="auto")
        assert resolved.num_instances == 10

    def test_auto_total_emails_rounds_up(self):
        cs = CorpusSize(total_emails=43)
        resolved = resolve_corpus_size(cs, num_plot_points=7, blur_level="medium", mode="auto")
        # 42 epi -> ceil(43/42) = 2
        assert resolved.num_instances == 2

    def test_auto_exact_multiple(self):
        cs = CorpusSize(total_emails=84)
        resolved = resolve_corpus_size(cs, num_plot_points=7, blur_level="medium", mode="auto")
        # 42 epi -> ceil(84/42) = 2
        assert resolved.num_instances == 2

    def test_auto_neither_total_nor_explicit_raises(self):
        cs = CorpusSize()  # neither total_emails nor num_instances set
        with pytest.raises(ValueError, match="must provide either"):
            resolve_corpus_size(cs, num_plot_points=7, blur_level="medium", mode="auto")

    def test_manual_requires_num_instances(self):
        cs = CorpusSize(total_emails=500)  # num_instances not set
        with pytest.raises(ValueError, match="num_instances is required"):
            resolve_corpus_size(cs, num_plot_points=7, blur_level="medium", mode="manual")

    def test_manual_total_emails_logs_warning(self, caplog):
        import logging
        cs = CorpusSize(total_emails=999, num_instances=5)
        with caplog.at_level(logging.WARNING):
            resolve_corpus_size(cs, num_plot_points=7, blur_level="medium", mode="manual")
        assert any("total_emails" in r.message and "ignored" in r.message for r in caplog.records)
