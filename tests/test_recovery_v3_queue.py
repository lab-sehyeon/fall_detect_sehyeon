import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import run_recovery_v3_queue as queue


class QueueTests(unittest.TestCase):
    def test_failed_pilot_never_passes_queue(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'report.json';queue.save(path,{'completed':True,'selected':None,'baseline_equality_passed':True})
            with self.assertRaisesRegex(RuntimeError,'not selected'): queue.verify_report(path,pilot=True)

    def test_full_requires_research_gate_and_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'final_report.json';queue.save(root/'independent_audit.json',{'passed':True})
            record={'passed':True,'research_usable':False,'audit_sha256':queue.sha(root/'independent_audit.json')};queue.save(path,record)
            with self.assertRaises(RuntimeError): queue.verify_report(path)
            self.assertEqual(queue.verify_report(path,smoke=True),queue.sha(path))
            record['research_usable']=True;queue.save(path,record);queue.save(root/'independent_audit.json',{'passed':False})
            with self.assertRaisesRegex(RuntimeError,'audit'): queue.verify_report(path)

    def test_dependency_failure_and_pause_are_terminal(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(queue,'DEPENDENCY',Path(folder)),patch.object(queue,'QUEUE',Path(folder)):
            queue.save(Path(folder)/'v1/status.json',{'stage':'failed'})
            with self.assertRaisesRegex(RuntimeError,'dependency failed'): queue.dependency_ready()
            (Path(folder)/'PAUSE_REQUESTED').touch()
            with self.assertRaisesRegex(RuntimeError,'pause requested'): queue.pause()


if __name__=='__main__': unittest.main()
