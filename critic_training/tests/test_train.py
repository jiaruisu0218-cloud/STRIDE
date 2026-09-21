import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import configuration


class TrainingConfigurationTests(unittest.TestCase):
    def test_dpo_reference_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'adapter_config.json').write_text('{}', encoding='utf-8')
            (root/'pairs.json').write_text(json.dumps([
                dict(messages=[], chosen={}, rejected={})]), encoding='utf-8')
            info = dict(critic_preferences=dict(file_name='pairs.json', ranking=True))
            (root/'dataset_info.json').write_text(json.dumps(info), encoding='utf-8')
            args = argparse.Namespace(stage='dpo', model='local-model', data_dir=root,
                                      output_dir=root/'output', sft_adapter=root)
            config = configuration(args)
            self.assertEqual(config['model_name_or_path'], config['ref_model'])
            self.assertEqual(config['adapter_name_or_path'], config['ref_model_adapters'])
            self.assertFalse(config['create_new_adapter'])
            args.output_dir.mkdir()
            (args.output_dir/'checkpoint').write_text('existing', encoding='utf-8')
            with self.assertRaises(ValueError):
                configuration(args)

    def test_sft_requires_development_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'train.json').write_text('[{"messages": []}]', encoding='utf-8')
            (root/'dev.json').write_text('[]', encoding='utf-8')
            info = dict(critic_train=dict(file_name='train.json'),
                        critic_dev=dict(file_name='dev.json'))
            (root/'dataset_info.json').write_text(json.dumps(info), encoding='utf-8')
            args = argparse.Namespace(stage='sft', model='local-model', data_dir=root,
                                      output_dir=root/'output', sft_adapter=None)
            with self.assertRaises(ValueError):
                configuration(args)


if __name__ == '__main__':
    unittest.main()
