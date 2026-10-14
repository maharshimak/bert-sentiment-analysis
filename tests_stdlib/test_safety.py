"""No third-party packages, models, database, training or network required."""
import contextlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src import metrics, inference, model, wait_for_db


class SafetyTests(unittest.TestCase):
    def test_accuracy_and_exact_binary_f1(self):
        self.assertEqual(metrics.compute_accuracy([0,1,1,0],[0,1,0,0]),0.75)
        self.assertAlmostEqual(metrics.compute_f1_binary([1,1,0,1,0],[1,0,1,1,0]),2/3)
        self.assertEqual(metrics.compute_accuracy([],[]),0.0)
        self.assertEqual(metrics.compute_f1_binary([],[]),0.0)

    def test_metrics_reject_mismatch_and_nonbinary(self):
        for function in (metrics.compute_accuracy,metrics.compute_f1_binary):
            with self.assertRaises(ValueError):function([0,1],[0])
            with self.assertRaises(ValueError):function([2],[0])

    def test_import_and_default_inference_are_inert(self):
        with self.assertRaises(RuntimeError):inference.predict('synthetic example')
        for name in ('torch','transformers','psycopg2'):
            self.assertNotIn(name,sys.modules)

    def test_model_factories_are_local_and_remote_code_disabled(self):
        tokenizer=MagicMock();classifier=MagicMock()
        fake=types.SimpleNamespace(AutoTokenizer=tokenizer,AutoModelForSequenceClassification=classifier)
        with patch.dict(sys.modules,{'transformers':fake}):
            model.get_tokenizer();model.get_model()
        self.assertTrue(tokenizer.from_pretrained.call_args.kwargs['local_files_only'])
        self.assertFalse(classifier.from_pretrained.call_args.kwargs['trust_remote_code'])
        self.assertTrue(classifier.from_pretrained.call_args.kwargs['use_safetensors'])

    def test_mocked_prediction_requires_only_explicit_execution(self):
        fake_torch=types.SimpleNamespace(no_grad=contextlib.nullcontext,argmax=lambda *a,**k:types.SimpleNamespace(item=lambda:1))
        tokenizer=MagicMock(return_value={'input_ids':MagicMock()})
        classifier=MagicMock();logits=object();classifier.return_value=types.SimpleNamespace(logits=logits)
        with patch.dict(sys.modules,{'torch':fake_torch}),patch.object(inference,'get_model',return_value=classifier),patch.object(inference,'get_tokenizer',return_value=tokenizer):
            label,actual=inference.predict('synthetic example',allow_inference=True)
        self.assertEqual(label,1);self.assertIs(actual,logits)
        self.assertEqual(tokenizer.call_args.kwargs['max_length'],512)

    def test_database_default_never_connects_and_errors_hide_values(self):
        connect=MagicMock(side_effect=RuntimeError('private connection detail'))
        with self.assertRaises(RuntimeError):wait_for_db.wait_for_db(connect=connect)
        connect.assert_not_called()
        environment={'DB_USER':'fixture','DB_PASSWORD':'fixture','DB_NAME':'fixture'}
        with patch.dict(os.environ,environment),self.assertRaises(RuntimeError) as error:
            wait_for_db.wait_for_db(allow_connection=True,attempts=1,interval=0,connect=connect)
        self.assertNotIn('private connection detail',str(error.exception))

    def test_cli_default_refuses_without_loading_models(self):
        result=subprocess.run([sys.executable,'-B','-m','src.inference'],cwd=ROOT,capture_output=True,text=True,timeout=5)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('--allow-inference',result.stderr)


if __name__=='__main__':unittest.main()
