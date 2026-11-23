"""Exercise installer control flow without downloading packages or changing Conda."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/setup/setup_mamma_env.sh"


class MammaSetupTests(unittest.TestCase):
    def run_fixture(self, *, existing_env=False, fail_pip=False, incomplete=False, profile='legacy', sdf_dispatch='triangles.type()', detectron_bound='iopath>=0.1.7,<0.1.10', fail_runtime_deps=False, fail_visualization=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkout = root / "existing mamma"
            files = {
                "inference/cli/run.py": "# preserved local checkout\n",
                "segmentation/core/pipeline.py": "# segmentation import fixture\n",
                "requirements/mamma_conda.yaml": "name: mamma\n# torch==2.5.1 torchvision==0.20.1 /whl/cu124\n",
                "requirements/requirements.txt": "smplx\ngit+https://github.com/facebookresearch/sam2\ngit+https://github.com/facebookresearch/sam3\n",
                "requirements/requirements_no_build_isolation.txt": "git+https://github.com/facebookresearch/detectron2.git\ngit+https://github.com/cuevhv/pytorch_sdf@torch2.5-cu124\n",
            }
            for relative, content in files.items():
                path = checkout / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            if incomplete:
                (checkout / "inference/cli/run.py").unlink()
            bin_dir = root / "bin"
            bin_dir.mkdir()
            # Absolute interpreter avoids relying on PATH resolution in the fake tools.
            import sys
            fake = bin_dir / "conda"
            fake.write_text(
                f"#!{sys.executable}\n"
                "import json, os, pathlib, subprocess, sys\n"
                "args = sys.argv[1:]\n"
                "with open(os.environ['CALL_LOG'], 'a') as f: f.write(json.dumps(args) + '\\n')\n"
                "if 'python' in args and args[args.index('python') + 1] == '-':\n"
                "    sys.exit(subprocess.run([sys.executable] + args[args.index('python') + 1:]).returncode)\n"
                "if 'pip' in args and 'check' in args:\n"
                "    assert pathlib.Path(os.environ['DEPS_READY']).exists(), 'dependencies were skipped'\n"
                "if any('MAMMA visualization imports passed:' in a for a in args):\n"
                "    if os.environ['FAIL_VISUALIZATION'] == '1': sys.exit(11)\n"
                "if args == ['env', 'list']:\n"
                "    print('mamma /fake/mamma' if os.environ['EXISTING_ENV'] == '1' else 'base /fake/base')\n"
                "if 'import sys; print(sys.prefix)' in args: print('/fake/mamma')\n"
                "if args[:2] in (['env', 'create'], ['env', 'update']) and os.environ['MAMMA_CUDA_PROFILE'] == 'blackwell':\n"
                "    recipe = pathlib.Path(args[args.index('-f') + 1]).read_text()\n"
                "    assert 'torch==2.7.1' in recipe and 'torchvision==0.22.1' in recipe and '/whl/cu128' in recipe\n"
                "if 'pip' in args and 'install' in args:\n"
                "    if os.environ['FAIL_PIP'] == '1': sys.exit(9)\n"
                "    if 'detectron2' in args and 'pytorch_sdf' in args:\n"
                "        assert '--no-deps' not in args and '--force-reinstall' not in args\n"
                "        assert '-c' in args\n"
                "        if os.environ['FAIL_RUNTIME_DEPS'] == '1': sys.exit(10)\n"
                "        pathlib.Path(os.environ['DEPS_READY']).touch()\n"
                "    if '-r' not in args: sys.exit(0)\n"
                "    req = pathlib.Path(args[args.index('-r') + 1])\n"
                "    if req.name == 'requirements.txt':\n"
                "        assert 'facebookresearch/sam3' not in req.read_text()\n"
                "        assert 'facebookresearch/sam2' in req.read_text()\n"
                "        expected = 'torch==2.7.1' if os.environ['MAMMA_CUDA_PROFILE'] == 'blackwell' else 'torch==2.5.1'\n"
                "        assert expected in pathlib.Path(args[args.index('-c') + 1]).read_text()\n"
                "        assert 'pyarrow==19.0.1' in pathlib.Path(args[args.index('-c') + 1]).read_text()\n"
                "    if req.name == 'detectron_requirements.txt':\n"
                "        setup = pathlib.Path(req.read_text().splitlines()[0]) / 'setup.py'\n"
                "        assert '\"iopath==0.1.10\"' in setup.read_text()\n"
                "    if req.name == 'build_requirements.txt':\n"
                "        lines = req.read_text().splitlines()\n"
                "        assert len(lines) == 2\n"
                "        setup = pathlib.Path(lines[0]) / 'setup.py'\n"
                "        assert '\"iopath==0.1.10\"' in setup.read_text()\n"
                "        source = pathlib.Path(lines[1]) / 'src/bvh_cuda_op.cu'\n"
                "        assert 'triangles.scalar_type()' in source.read_text()\n"
                "        assert 'triangles.type()' not in source.read_text()\n"
                "        assert '--force-reinstall' in args and '--no-deps' in args\n"
            )
            fake.chmod(0o755)
            # Only temporary dependency clones are allowed; preserve MAMMA.
            git = bin_dir / "git"
            git.write_text(
                f"#!{sys.executable}\n"
                "import os, pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "if args[:3] == ['clone', '--no-checkout', 'https://github.com/cuevhv/pytorch_sdf']:\n"
                "    source = pathlib.Path(args[3]) / 'src/bvh_cuda_op.cu'\n"
                "    source.parent.mkdir(parents=True)\n"
                "    source.write_text('AT_DISPATCH_FLOATING_TYPES(' + os.environ['SDF_DISPATCH'] + ', \"bvh_tree_building\", body);\\n')\n"
                "elif args[:3] == ['clone', '--no-checkout', 'https://github.com/facebookresearch/detectron2']:\n"
                "    source = pathlib.Path(args[3]) / 'setup.py'\n"
                "    source.parent.mkdir(parents=True)\n"
                "    source.write_text('install_requires=[\"' + os.environ['DETECTRON_BOUND'] + '\"]\\n')\n"
                "elif args[0] == '-C' and pathlib.Path(args[1]).name == 'detectron2':\n"
                "    assert args[2:] == ['checkout', '--detach', '1e3e13bbf607b54f62205c4c33922521822fb298']\n"
                "elif args[0] == '-C' and pathlib.Path(args[1]).name == 'pytorch_sdf':\n"
                "    assert args[2:] in (['checkout', '--detach', '58ff57b542a998e406327b94881bf6f7ce13a390'], ['submodule', 'update', '--init', '--recursive'])\n"
                "else: sys.exit(77)\n"
            )
            git.chmod(0o755)
            log = root / "calls.jsonl"
            env = os.environ | {
                "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
                "CALL_LOG": str(log),
                "EXISTING_ENV": str(int(existing_env)),
                "FAIL_PIP": str(int(fail_pip)),
                "MAMMA_CUDA_PROFILE": profile,
                "SDF_DISPATCH": sdf_dispatch,
                "DETECTRON_BOUND": detectron_bound,
                "DEPS_READY": str(root / 'deps-ready'),
                "FAIL_RUNTIME_DEPS": str(int(fail_runtime_deps)),
                "FAIL_VISUALIZATION": str(int(fail_visualization)),
            }
            result = subprocess.run(
                [str(SCRIPT), "--mamma-repo", checkout.name],
                cwd=root, env=env, capture_output=True, text=True,
            )
            for relative, content in files.items():
                if incomplete and relative == "inference/cli/run.py":
                    continue
                self.assertEqual((checkout / relative).read_text(), content)
            calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            return result, calls

    def test_create_preserves_checkout_and_handles_spaces(self):
        result, calls = self.run_fixture()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(any(c[:2] == ["env", "create"] for c in calls))
        self.assertTrue(any("check" in c and "pip" in c for c in calls))
        self.assertIn("Download model assets next", result.stdout)

    def test_visualization_wheel_repair_precedes_import_and_pip_check(self):
        result, calls = self.run_fixture(existing_env=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        wheel = next(i for i, c in enumerate(calls) if 'pyarrow==19.0.1' in c)
        for option in ('--force-reinstall', '--no-deps', '--only-binary=:all:'):
            self.assertIn(option, calls[wheel])
        imports = next(i for i, c in enumerate(calls)
                       if any('MAMMA visualization imports passed:' in a for a in c))
        check = next(i for i, c in enumerate(calls) if 'pip' in c and 'check' in c)
        self.assertLess(wheel, imports)
        self.assertLess(imports, check)

    def test_broken_visualization_stops_before_success(self):
        result, calls = self.run_fixture(fail_visualization=True)
        self.assertEqual(result.returncode, 11, result.stderr)
        self.assertNotIn('MAMMA environment installed', result.stdout)
        self.assertFalse(any('check' in c and 'pip' in c for c in calls))

    def test_blackwell_changes_recipe_and_rebuilds_extensions(self):
        result, calls = self.run_fixture(existing_env=True, profile='blackwell')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(any('cuda-toolkit=12.8' in c for c in calls))
        self.assertTrue(any('torch==2.7.1' in c and 'https://download.pytorch.org/whl/cu128' in c for c in calls))
        self.assertTrue(any('--force-reinstall' in c and '--no-deps' in c and '--no-build-isolation' in c for c in calls))

    def test_blackwell_rejects_unexpected_sdf_source_before_build(self):
        result, calls = self.run_fixture(profile='blackwell', sdf_dispatch='changed.type()')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('compatibility patch aborted', result.stderr)
        self.assertFalse(any('--no-build-isolation' in c for c in calls))
        self.assertFalse(any('check' in c and 'pip' in c for c in calls))

    def test_detectron_patch_rejects_unexpected_dependency_bound(self):
        result, calls = self.run_fixture(detectron_bound='iopath>=0.2')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('compatibility patch aborted', result.stderr)
        self.assertFalse(any('--no-build-isolation' in c for c in calls))

    def test_runtime_dependency_failure_stops_before_check_and_success(self):
        result, calls = self.run_fixture(profile='blackwell', fail_runtime_deps=True)
        self.assertEqual(result.returncode, 10, result.stderr)
        self.assertNotIn('MAMMA environment installed', result.stdout)
        self.assertFalse(any('check' in c and 'pip' in c for c in calls))

    def test_rerun_updates_environment_without_pruning(self):
        result, calls = self.run_fixture(existing_env=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(any(c[:2] == ["env", "update"] for c in calls))
        self.assertFalse(any("--prune" in c for c in calls))
        self.assertFalse(any(c[:2] == ["env", "create"] for c in calls))

    def test_dependency_failure_stops_before_success(self):
        result, calls = self.run_fixture(fail_pip=True)
        self.assertEqual(result.returncode, 9, result.stderr)
        self.assertNotIn("MAMMA environment installed", result.stdout)
        self.assertFalse(any("check" in c and "pip" in c for c in calls))

    def test_incomplete_checkout_does_not_mutate_environment(self):
        result, calls = self.run_fixture(incomplete=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("invalid or incomplete", result.stderr)
        self.assertEqual(calls, [])

    def test_missing_argument_has_actionable_error(self):
        result = subprocess.run([str(SCRIPT), "--mamma-repo"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires a directory", result.stderr)
