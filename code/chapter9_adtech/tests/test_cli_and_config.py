"""Chapter 9 CLI and config-identity checks.

No training run and no dataset files.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

CHAPTER = Path(__file__).resolve().parents[1]
if str(CHAPTER) not in sys.path:
    sys.path.insert(0, str(CHAPTER))


def _run(script, *args, cwd=None):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252"
    env.pop("PYTHONUTF8", None)
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        env=env,
        cwd=str(cwd or CHAPTER),
    )


class FocalLossFlagTests(unittest.TestCase):
    def _parse(self, *extra):
        old = sys.argv
        sys.argv = ["train_esmm.py", *extra]
        try:
            from train_esmm import parse_args
            return parse_args()
        finally:
            sys.argv = old

    def test_focal_loss_defaults_on(self):
        self.assertIs(self._parse().use_focal_loss, True)

    def test_focal_loss_can_be_turned_off(self):
        self.assertIs(self._parse("--no-use_focal_loss").use_focal_loss, False)

    def test_legacy_use_focal_loss_flag_still_parses(self):
        self.assertIs(self._parse("--use_focal_loss").use_focal_loss, True)


class HelpEncodingTests(unittest.TestCase):
    def _assert_ascii_help(self, script, cwd):
        result = _run(script, "--help", cwd=cwd)
        stderr = result.stderr.decode("cp1252", errors="replace")
        self.assertEqual(result.returncode, 0, stderr)
        text = result.stdout.decode("cp1252")
        text.encode("ascii")
        return text

    def test_preprocess_help_is_ascii_under_cp1252(self):
        script = CHAPTER / "scripts" / "preprocess_ali_ccp.py"
        text = self._assert_ascii_help(script, script.parent)
        self.assertIn("--pct", text)
        self.assertIn("->", text)
        self.assertNotIn("\u2192", text)

    def test_train_esmm_help_offers_both_focal_flags(self):
        text = self._assert_ascii_help(CHAPTER / "train_esmm.py", CHAPTER)
        self.assertIn("--use_focal_loss", text)
        self.assertIn("--no-use_focal_loss", text)


class ConfigIdentityTests(unittest.TestCase):
    def test_config_classes_are_the_same_objects(self):
        import config
        import models
        import models.esmm as esmm
        import models.split_nn as split_nn
        import train_esmm
        import train_split_nn

        self.assertIs(config.ESMMConfig, esmm.ESMMConfig)
        self.assertIs(config.ESMMConfig, models.ESMMConfig)
        self.assertIs(config.ESMMConfig, train_esmm.ESMMConfig)
        self.assertIs(config.SplitNNConfig, split_nn.SplitNNConfig)
        self.assertIs(config.SplitNNConfig, models.SplitNNConfig)
        self.assertIs(config.SplitNNConfig, train_split_nn.SplitNNConfig)
        self.assertIs(config.ESMMConfig().use_focal_loss, True)
        self.assertIs(config.SplitNNConfig().use_focal_loss, False)

    def test_importing_models_before_config_does_not_cycle(self):
        code = "\n".join([
            "import models.esmm",
            "import models.split_nn",
            "import config",
            "import models",
            "assert config.ESMMConfig is models.esmm.ESMMConfig is models.ESMMConfig",
            "assert config.SplitNNConfig is models.split_nn.SplitNNConfig is models.SplitNNConfig",
        ])
        env = os.environ.copy()
        env["PYTHONPATH"] = str(CHAPTER)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            cwd=str(CHAPTER),
            env=env,
        )
        message = result.stderr.decode("utf-8", errors="replace")
        self.assertEqual(result.returncode, 0, message)


if __name__ == "__main__":
    unittest.main()
