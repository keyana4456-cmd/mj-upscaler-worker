import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import engine

class PerformanceControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = str(Path(self.tmp.name)/'input.mp4')
        Path(self.src).write_bytes(b'test')
        self.scope = patch.dict(os.environ, {'MJU_WORK': self.tmp.name})
        self.scope.start(); self.addCleanup(self.scope.stop)

    def job(self, **kw):
        with patch.object(engine, 'probe', return_value=dict(w=446,h=336,fps=30,frames=60,dur=2)), patch.object(engine, 'crop_detect', return_value=None):
            return engine.Job(self.src, self.tmp.name, **kw)

    def test_heavy_default_preserved(self):
        self.assertTrue(self.job(mode=engine.HEAVY).face)

    def test_explicit_face_disable_applies_to_heavy(self):
        j=self.job(mode=engine.HEAVY,face_override=False)
        self.assertFalse(j.face)
        self.assertNotEqual(j.work,self.job(mode=engine.HEAVY).work)

    def test_face_override_rejects_string(self):
        with self.assertRaises(ValueError): self.job(face_override='false')

    def test_cloud_command_keeps_acceleration_controls(self):
        j=self.job(quality='Pro 7B')
        captured=[]
        class Captured(Exception): pass
        def capture(cmd): captured.extend(cmd); raise Captured
        options=dict(NO_OFFLOAD=True,BIG_GPU=False,BLOCKS_TO_SWAP=0,SEED_BATCH=13,
                     ATTENTION='sageattn_2',COLOR_CORRECTION='none',CLOUD_SEED_RES=1080)
        with patch.multiple(engine, **options),patch.object(engine,'run'),patch.object(j,'_run_seed',side_effect=capture):
            with self.assertRaises(Captured): j._upscale(self.tmp.name,self.tmp.name,60)
        self.assertNotIn('--dit_offload_device',captured)
        self.assertNotIn('--blocks_to_swap',captured)
        for flag,value in [('--attention_mode','sageattn_2'),('--batch_size','13'),('--color_correction','none')]:
            self.assertEqual(captured[captured.index(flag)+1],value)

if __name__=='__main__': unittest.main()
