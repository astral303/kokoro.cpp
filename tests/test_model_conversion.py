#!/usr/bin/env python3
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ConversionManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(
            (ROOT / "models" / "languages.json").read_text(encoding="utf-8")
        )

    def test_pins_official_revision(self):
        self.assertEqual(self.manifest["source"], "hexgrad/Kokoro-82M")
        self.assertEqual(
            self.manifest["revision"],
            "f3ff3571791e39611d31c381e3a41a3af07b4987",
        )
        self.assertEqual(
            self.manifest["checkpoint_sha256"],
            "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4",
        )

    def test_lists_all_official_voices_once(self):
        voices = [
            voice
            for language in self.manifest["languages"]
            for voice in language["voices"]
        ]
        self.assertEqual(len(voices), 54)
        self.assertEqual(len(set(voices)), 54)
        self.assertEqual(
            set(voices),
            {
                "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
                "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah",
                "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
                "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
                "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel",
                "bm_fable", "bm_george", "bm_lewis", "ef_dora", "em_alex",
                "em_santa", "ff_siwis", "hf_alpha", "hf_beta", "hm_omega",
                "hm_psi", "if_sara", "im_nicola", "jf_alpha",
                "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
                "pf_dora", "pm_alex", "pm_santa", "zf_xiaobei", "zf_xiaoni",
                "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi",
                "zm_yunxia", "zm_yunyang",
            },
        )

    def test_language_codes_are_complete_and_unique(self):
        languages = self.manifest["languages"]
        self.assertEqual(len(languages), 9)
        self.assertEqual(
            {language["kokoro_code"] for language in languages},
            set("abefhijpz"),
        )
        self.assertEqual(
            {language["espeak_voice"] for language in languages},
            {"en-us", "en-gb", "es", "fr-fr", "hi", "it", "ja", "pt-br", "cmn"},
        )

    def test_voice_prefix_matches_language(self):
        for language in self.manifest["languages"]:
            code = language["kokoro_code"]
            for voice in language["voices"]:
                self.assertTrue(voice.startswith(code), (language["name"], voice))


if __name__ == "__main__":
    unittest.main()
