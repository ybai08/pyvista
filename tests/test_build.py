"""Tests for the backend build engine and the manifest machinery behind it."""

from __future__ import annotations

from dataclasses import replace
import json
import os
import re
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
from rich.console import Console

from pyvista._build import BuildConfig
from pyvista._build import Builder
from pyvista._build import BuildError
from pyvista._build import ManifestError
from pyvista._build import load_manifest
from pyvista._build import phases
from pyvista._build import tools
from pyvista._build.manifest import Option
from pyvista._build.manifest import cmake_entries
from pyvista._build.manifest import cmake_unset
from pyvista._build.manifest import parse_manifest
from pyvista._build.manifest import resolve_groups
from pyvista._build.manifest import resolve_options
from pyvista._build.registry import find_manifest
from pyvista._build.registry import known_backends
from pyvista._build.registry import shipped_manifest
from pyvista.core._vtk_utilities import _MAX_SUPPORTED_VTK_VERSION
from pyvista.core._vtk_utilities import VersionInfo

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

OS_NAMES = ['linux', 'macos', 'windows']

MINIMAL_MANIFEST = {
    'manifest_version': 1,
    'backend': {'name': 'demo', 'import_name': 'demo', 'default_ref': 'v1.0'},
    'groups': {
        'a': {'help': 'A', 'default': True, 'cmake': ['A=ON'], 'cmake_off': ['A=OFF']},
        'b': {'help': 'B', 'default': False, 'cmake': ['B=ON'], 'conflicts': ['c']},
        'c': {
            'help': 'C',
            'default': False,
            'cmake': ['C=ON'],
            'conflicts': ['b'],
            'os': ['linux'],
        },
        'core': {'help': 'Core', 'required': True, 'cmake': ['CORE=ON']},
    },
    'options': {
        'lto': {'help': 'LTO', 'default': False, 'cmake': 'LTO={value}'},
        'smp': {
            'help': 'SMP',
            'default': 'STDThread',
            'cmake': 'SMP={value}',
            'choices': ['STDThread', 'TBB'],
        },
        'jobs': {'help': 'Jobs', 'default': 2, 'cmake': 'JOBS={value}'},
    },
    'hooks': {'post_build': ['{python} prune.py {build_dir}']},
}


@pytest.fixture
def demo():
    return parse_manifest(MINIMAL_MANIFEST)


def test_shipped_vtk_manifest_loads():
    path = shipped_manifest('vtk')
    assert path is not None
    manifest = load_manifest(path, default_ref='v1')
    assert manifest.name == 'vtk'
    assert manifest.import_name == 'vtkmodules'
    assert 'rendering' in manifest.groups
    assert manifest.groups['headless-egl'].os == ('linux',)
    assert find_manifest('vtk', None).path == path
    with pytest.raises(ManifestError, match='default_ref'):
        load_manifest(path)


def test_vtk_default_ref_is_max_supported():
    maximum = VersionInfo._format(_MAX_SUPPORTED_VTK_VERSION)
    assert find_manifest('vtk', None).default_ref == f'v{maximum}'


def test_load_manifest_default_ref(tmp_path: Path):
    path = tmp_path / 'pyvista-build.toml'
    path.write_text(
        'manifest_version = 1\n[backend]\nname = "x"\nimport_name = "x"\ndefault_ref = "r"\n'
    )
    assert load_manifest(path, default_ref='other').default_ref == 'r'
    path.write_text('manifest_version = 1\nbackend = 1\n')
    with pytest.raises(ManifestError, match='backend'):
        load_manifest(path, default_ref='other')


@pytest.mark.parametrize('os_name', OS_NAMES)
def test_vtk_manifest_defaults_match_ci_flags(os_name):
    manifest = find_manifest('vtk', None)
    groups = resolve_groups(manifest, os_name)
    entries = cmake_entries(
        manifest, os_name, groups=groups, options=resolve_options(manifest, None)
    )
    ci_flags = {
        'VTK_WHEEL_BUILD=ON',
        'VTK_WRAP_PYTHON=ON',
        'VTK_GROUP_ENABLE_Rendering=WANT',
        'VTK_BUILD_TESTING=OFF',
        'VTK_BUILD_DOCUMENTATION=OFF',
        'VTK_BUILD_EXAMPLES=OFF',
        'VTK_USE_PCH=OFF',
        'VTK_DISPATCH_SOA_ARRAYS=ON',
        'VTK_WRAP_SERIALIZATION=ON',
        'VTK_BUILD_PYI_FILES=ON',
        'VTK_MODULE_ENABLE_VTK_FiltersParallelDIY2=YES',
        'VTK_MODULE_ENABLE_VTK_RenderingMatplotlib=YES',
        'VTK_MODULE_ENABLE_VTK_IOImport=YES',
        'VTK_MODULE_ENABLE_VTK_IOParallelExodus=YES',
        'VTK_MODULE_ENABLE_VTK_IOXdmf2=YES',
        'VTK_MODULE_ENABLE_VTK_WebCore=YES',
        'VTK_MODULE_ENABLE_VTK_WebGLExporter=YES',
        'VTK_MODULE_ENABLE_VTK_WebPython=YES',
        'VTK_SMP_IMPLEMENTATION_TYPE=Sequential',
    }
    if os_name == 'macos':
        ci_flags.add('CMAKE_OSX_DEPLOYMENT_TARGET=11.0')
    assert set(entries) == ci_flags
    assert not any('EGL' in e or 'OSMESA' in e for e in entries)


def test_resolve_groups_defaults(demo):
    assert resolve_groups(demo, 'linux') == ['a', 'core']
    assert resolve_groups(demo, 'linux', minimal=True) == ['core']
    assert resolve_groups(demo, 'macos', all_=True) == ['a', 'b', 'core']
    assert resolve_groups(demo, 'linux', with_=['b'], without=['a']) == ['b', 'core']


def test_resolve_groups_errors(demo):
    assert resolve_groups(demo, 'linux', all_=True) == ['a', 'b', 'core']
    with pytest.raises(ManifestError, match='conflicts'):
        resolve_groups(demo, 'linux', with_=['b', 'c'])
    with pytest.raises(ManifestError, match='required'):
        resolve_groups(demo, 'linux', without=['core'])
    with pytest.raises(ManifestError, match='Unknown group'):
        resolve_groups(demo, 'linux', with_=['nope'])
    with pytest.raises(ManifestError, match='not available on macos'):
        resolve_groups(demo, 'macos', with_=['c'])
    with pytest.raises(ManifestError, match='mutually exclusive'):
        resolve_groups(demo, 'linux', all_=True, minimal=True)


def test_cmake_entries_use_off_list(demo):
    assert cmake_entries(demo, 'linux', groups=['core'], options={}) == ['A=OFF', 'CORE=ON']
    assert cmake_entries(demo, 'linux', groups=['a', 'core'], options={}) == ['A=ON', 'CORE=ON']


def test_resolve_options(demo):
    values = resolve_options(demo, ['lto=on', 'smp=TBB', 'jobs=8'])
    assert values == {'lto': True, 'smp': 'TBB', 'jobs': 8}
    entries = cmake_entries(demo, 'linux', groups=['core'], options=values)
    assert entries[-3:] == ['LTO=ON', 'SMP=TBB', 'JOBS=8']
    with pytest.raises(ManifestError, match='boolean'):
        resolve_options(demo, ['lto=maybe'])
    with pytest.raises(ManifestError, match='one of'):
        resolve_options(demo, ['smp=OpenMP'])
    with pytest.raises(ManifestError, match='integer'):
        resolve_options(demo, ['jobs=many'])
    with pytest.raises(ManifestError, match='Unknown option'):
        resolve_options(demo, ['nope=1'])
    with pytest.raises(ManifestError, match=re.escape("'lto' needs a value: write lto=VALUE")):
        resolve_options(demo, ['lto'])


@pytest.mark.parametrize(
    ('patch', 'match'),
    [
        ({'manifest_version': 2}, 'manifest_version'),
        ({'grups': {}}, 'unknown key'),
        ({'backend': {'name': 'x'}}, 'default_ref'),
        ({'backend': {'name': 'x', 'default_ref': 'r', 'url': 5}}, "'url' must be a string"),
        ({'backend': {'name': 'x', 'default_ref': 'r', 'colour': 1}}, 'unknown key'),
        (
            {'backend': {'name': 'x', 'default_ref': 'r', 'version_command': 'v {home}'}},
            'unknown placeholder',
        ),
        ({'groups': ['g']}, 'table'),
        ({'groups': {'g': 'x'}}, 'table'),
        ({'groups': {'g': {'default': True}}}, 'help'),
        ({'groups': {'g': {'help': 'x', 'defualt': True}}}, 'unknown key'),
        ({'groups': {'g': {'help': 'x', 'default': 'false'}}}, "'default' must be a boolean"),
        ({'groups': {'g': {'help': 'x', 'os': ['beos']}}}, 'unknown os'),
        ({'groups': {'g': {'help': 'x', 'conflicts': ['zz']}}}, 'unknown group'),
        (
            {'groups': {'g': {'help': 'x', 'conflicts': ['h']}, 'h': {'help': 'y'}}},
            "list 'g' in",
        ),
        ({'options': {'o': {'help': 'x', 'default': True, 'cmake': 'X=ON'}}}, 'placeholder'),
        (
            {'options': {'o': {'help': 'x', 'default': True, 'cmake': 5}}},
            "'cmake' must be a string",
        ),
        (
            {'options': {'o': {'help': 'x', 'default': True, 'cmake': 'X=${HOME}{value}'}}},
            'unknown placeholder',
        ),
        ({'options': {'o': {'help': 'x', 'default': [], 'cmake': 'X={value}'}}}, 'default'),
        (
            {
                'options': {
                    'o': {'help': 'x', 'default': 'C', 'cmake': 'X={value}', 'choices': ['A']}
                }
            },
            'not one of the choices',
        ),
        ({'hooks': {'pre_build': []}}, 'unknown hook'),
        ({'hooks': {'post_build': ['{interpreter} x.py']}}, 'unknown placeholder'),
        ({'hooks': {'post_build': ['{python x.py']}}, 'literal brace'),
        ({'options': {'o': 5}}, r'\[options.o\] must be a table'),
        (
            {
                'options': {
                    'o': {'help': 'x', 'default': 1, 'cmake': 'X={value}', 'choices': ['A']}
                }
            },
            'needs a string',
        ),
        ({'cmake': {'init_cache': {'beos': 'x'}}}, 'init_cache'),
        ({'cmake': {'init_cache': {'linux': 5}}}, 'init_cache'),
        ({'cmake': {'linx': ['A=1']}}, 'unknown key'),
    ],
)
def test_parse_manifest_errors(patch, match):
    data = dict(MINIMAL_MANIFEST, **patch)
    with pytest.raises(ManifestError, match=match):
        parse_manifest(data)


def test_load_manifest_bad_toml(tmp_path: Path):
    path = tmp_path / 'pyvista-build.toml'
    path.write_text('manifest_version = [')
    with pytest.raises(ManifestError, match=re.escape(str(path))):
        load_manifest(path)


def test_find_manifest_prefers_tree(tmp_path: Path):
    (tmp_path / 'pyvista-build.toml').write_text(
        'manifest_version = 1\n[backend]\nname = "vtk"\nimport_name = "x"\ndefault_ref = "r"\n'
    )
    assert find_manifest('vtk', tmp_path).import_name == 'x'
    with pytest.raises(ManifestError, match='ships no manifest'):
        find_manifest('nope', tmp_path.parent / 'missing')


def _config(tmp_path: Path, **kwargs) -> BuildConfig:
    defaults = dict(backend='vtk', build_dir=tmp_path / 'vtk-build', dry_run=True, os_name='linux')
    defaults.update(kwargs)
    return BuildConfig(**defaults)


@pytest.fixture
def no_tools(mocker: MockerFixture):
    """Pretend every tool is present and pip is available."""
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    mocker.patch.object(tools, 'executable_version', return_value='99.0')
    mocker.patch.object(tools, 'python_build_packages', return_value=[])


@pytest.mark.usefixtures('no_tools')
@pytest.mark.parametrize('os_name', OS_NAMES)
def test_dry_run_commands(tmp_path: Path, os_name):
    console = Console(record=True, width=4000, force_terminal=False)
    cfg = _config(tmp_path, os_name=os_name, defines=['FOO=1'], ref='v9.7.0', repair=True)
    builder = Builder(cfg, console)
    assert builder.build() is None
    out = console.export_text()
    assert out.startswith('Building backend: vtk')
    assert 'git clone --filter=blob:none https://gitlab.kitware.com/vtk/vtk.git' in out
    assert 'checkout v9.7.0' in out
    assert '-DFOO=1' in out
    assert '-DVTK_VERSION_SUFFIX= ' in out
    configure = next(line for line in out.splitlines() if ' -GNinja' in line)
    assert ('_PYTHON_HOST_PLATFORM=macosx-11.0-' in configure) == (os_name == 'macos')
    repairer = {
        'linux': 'auditwheel repair',
        'macos': 'delocate-wheel',
        'windows': 'delvewheel repair',
    }
    assert repairer[os_name] in out
    assert 'setup.py bdist_wheel' in out
    assert '--force-reinstall' in out  # printed as the install hint
    assert 'COMPILER_LAUNCHER' not in out


@pytest.mark.usefixtures('no_tools')
def test_dry_run_flags(tmp_path: Path):
    console = Console(record=True, width=4000, force_terminal=False)
    cfg = _config(
        tmp_path,
        minimal=True,
        build_type='Debug',
        jobs=3,
        ref='main',
        version_suffix='',
        options=['smp=TBB'],
        repair=False,
    )
    Builder(cfg, console).build()
    out = console.export_text()
    assert 'COMPILER_LAUNCHER' not in out
    assert '-DCMAKE_BUILD_TYPE=Debug' in out
    assert '--parallel 3' in out
    assert '-DVTK_GROUP_ENABLE_Rendering=NO' in out
    assert '-DVTK_SMP_IMPLEMENTATION_TYPE=TBB' in out
    assert '-DVTK_VERSION_SUFFIX= ' in out
    assert 'Skipped (--no-repair).' in out


@pytest.mark.usefixtures('no_tools')
def test_dry_run_dev_suffix(tmp_path: Path):
    console = Console(record=True, width=4000, force_terminal=False)
    Builder(_config(tmp_path, ref='main'), console).build()
    assert '-DVTK_VERSION_SUFFIX=dev0+g<sha>' in console.export_text()


def test_unknown_backend(tmp_path: Path):
    with pytest.raises(BuildError, match='Unknown backend'):
        Builder(_config(tmp_path, backend='nope'), Console(quiet=True)).build()


def test_missing_git(tmp_path: Path, mocker: MockerFixture):
    """A missing git stops at the tools step, before anything is cloned."""
    mocker.patch.object(tools.shutil, 'which', return_value=None)
    builder = Builder(_config(tmp_path), Console(quiet=True))
    source = mocker.spy(builder, 'phase_source')
    with pytest.raises(BuildError, match='git: Install git'):
        builder.build()
    assert not source.called
    with pytest.raises(BuildError, match='git was not found'):
        builder.phase_source()


def test_blocking_tool(tmp_path: Path, mocker: MockerFixture):
    mocker.patch.object(
        tools.shutil,
        'which',
        side_effect=lambda name, **_: (
            None if name in ('c++', 'clang++', 'g++') else f'/usr/bin/{name}'
        ),
    )
    mocker.patch.object(tools, 'executable_version', return_value='99.0')
    with pytest.raises(BuildError, match='build-essential'):
        Builder(_config(tmp_path), Console(quiet=True)).build()


def test_pip_tools_offered(tmp_path: Path, mocker: MockerFixture):
    missing = {'cmake', 'auditwheel'}
    mocker.patch.object(
        tools.shutil,
        'which',
        side_effect=lambda name, **_: None if name in missing else f'/usr/bin/{name}',
    )
    mocker.patch.object(tools, 'executable_version', return_value='99.0')
    mocker.patch.object(tools, 'python_build_packages', return_value=['wheel'])
    mocker.patch.object(
        tools, 'pip_install_command', side_effect=lambda *s, args=(): ['pip', 'install', *args, *s]
    )
    console = Console(record=True, width=4000, force_terminal=False)
    Builder(_config(tmp_path, repair=True), console).build()
    out = console.export_text()
    bullets = [f'  • {spec} into {sys.prefix}' for spec in ('cmake>=3.22', 'auditwheel', 'wheel')]
    assert '\n'.join(['The build installs:', *bullets]) in out
    assert re.search(r'wheel \(python\)\s+│ -\s+│ missing', out)
    assert re.search(r'setuptools \(python\)\s+│ .*│ ok', out)
    assert f'$ {phases._join(["pip", "install", "cmake>=3.22", "auditwheel", "wheel"])}' in out


def test_too_old_tool(tmp_path: Path, mocker: MockerFixture):
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    mocker.patch.object(tools, 'executable_version', return_value='3.10')
    mocker.patch.object(tools, 'python_build_packages', return_value=[])
    mocker.patch.object(
        tools, 'pip_install_command', side_effect=lambda *s, **_: ['pip', 'install', *s]
    )
    console = Console(record=True, width=4000, force_terminal=False)
    Builder(_config(tmp_path), console).build()
    out = console.export_text()
    assert 'too old' in out
    assert f'$ {phases._join(["pip", "install", "cmake>=3.22"])}' in out


@pytest.mark.usefixtures('no_tools')
def test_hooks_and_init_cache(tmp_path: Path):
    src = tmp_path / 'src'
    (src / '.git').mkdir(parents=True)
    (src / 'pyvista-build.toml').write_text(
        'manifest_version = 1\n'
        '[backend]\nname = "demo"\nimport_name = "demo"\ndefault_ref = "v1"\n'
        '[cmake]\ninit_cache = { linux = "ci/linux.cmake" }\n'
        '[hooks]\npost_build = ["{python} prune.py {build_dir}"]\n'
    )
    console = Console(record=True, width=4000, force_terminal=False)
    Builder(_config(tmp_path, backend='demo', source=src), console).build()
    out = console.export_text()
    assert f'-C {src / "ci" / "linux.cmake"}' in out
    assert f'prune.py {tmp_path / "vtk-build" / "build"}' in out
    assert 'git clone' not in out


def _fake_git(head: str, status: str = ' M file.cxx'):
    """Return a ``subprocess.run`` stand-in for a checkout at ``head`` with ``status`` output."""

    def run(cmd, **_):
        out = {'status': status, 'rev-parse': head if cmd[-1] == 'HEAD' else 'b' * 40}
        return subprocess.CompletedProcess(cmd, 0, stdout=out.get(cmd[1], ''), stderr='')

    return run


def test_existing_checkout_dirty(tmp_path: Path, mocker: MockerFixture):
    src = tmp_path / 'vtk-build' / 'src'
    (src / '.git').mkdir(parents=True)
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    mocker.patch.object(subprocess, 'run', side_effect=_fake_git('a' * 40))
    with pytest.raises(BuildError, match='local changes'):
        Builder(_config(tmp_path, ref='v9.7.0'), Console(quiet=True)).build()


@pytest.mark.usefixtures('no_tools')
def test_existing_checkout_at_ref_keeps_edits(tmp_path: Path, mocker: MockerFixture):
    src = tmp_path / 'vtk-build' / 'src'
    (src / '.git').mkdir(parents=True)
    run = mocker.patch.object(subprocess, 'run', side_effect=_fake_git('b' * 40))
    console = Console(record=True, width=4000, force_terminal=False)
    Builder(_config(tmp_path, ref='v9.7.0'), console).build()
    assert not any(c.args[0][1] == 'status' for c in run.call_args_list)
    assert 'checkout v9.7.0' not in console.export_text()


def test_state_change_prompts_reconfigure(tmp_path: Path, mocker: MockerFixture):
    cmake_dir = tmp_path / 'vtk-build' / 'build'
    cmake_dir.mkdir(parents=True)
    (cmake_dir / 'CMakeCache.txt').write_text('')
    (cmake_dir / '.pyvista-build.json').write_text('{"groups": []}')
    src = tmp_path / 'vtk-build' / 'src'
    (src / '.git').mkdir(parents=True)
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    mocker.patch.object(tools, 'executable_version', return_value='99.0')
    mocker.patch.object(tools, 'python_build_packages', return_value=[])
    console = Console(record=True, width=4000, force_terminal=False)
    Builder(_config(tmp_path), console).build()
    assert 'differs from the one this tree was set up with' in console.export_text()


def test_pip_install_command(mocker: MockerFixture):
    mocker.patch.object(tools.importlib.util, 'find_spec', return_value=object())
    cmd = tools.pip_install_command('cmake', args=('--no-deps',))
    assert cmd[1:] == ['-m', 'pip', 'install', '--no-deps', 'cmake']
    mocker.patch.object(tools.importlib.util, 'find_spec', return_value=None)
    mocker.patch.object(tools.shutil, 'which', return_value=None)
    assert tools.pip_install_command('cmake') is None


def test_env_options_and_version_command(tmp_path: Path, mocker: MockerFixture):
    src = tmp_path / 'src'
    (src / '.git').mkdir(parents=True)
    (src / 'pyvista-build.toml').write_text(
        'manifest_version = 1\n'
        '[backend]\nname = "demo"\nimport_name = "demo"\ndefault_ref = "v1"\n'
        'version_command = "{python} version.py"\n'
        '[options.lto]\nhelp = "LTO"\ndefault = false\nenv = "DEMO_LTO"\n'
        '[options.abi3]\nhelp = "abi3"\ndefault = true\ncmake = "DEMO_ABI3={value}"\n'
        '[hooks]\npost_configure = ["{python} check.py"]\n'
    )
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    mocker.patch.object(tools, 'executable_version', return_value='99.0')
    mocker.patch.object(tools, 'python_build_packages', return_value=[])
    console = Console(record=True, width=4000, force_terminal=False)
    Builder(_config(tmp_path, backend='demo', source=src, options=['lto=1']), console).build()
    out = console.export_text()
    assert '$ DEMO_LTO=1 cmake' in out
    assert re.search(r'\$ DEMO_LTO=1 \S+ check\.py', out)
    assert '-DDEMO_ABI3=ON' in out
    assert '-DDEMO_LTO' not in out
    assert '-DVTK_VERSION_SUFFIX=<version_command>' in out


def test_option_needs_cmake_or_env():
    base = {'help': 'x', 'default': True}
    with pytest.raises(ManifestError, match='exactly one'):
        parse_manifest(dict(MINIMAL_MANIFEST, options={'o': base}))
    with pytest.raises(ManifestError, match='exactly one'):
        parse_manifest(
            dict(MINIMAL_MANIFEST, options={'o': dict(base, cmake='X={value}', env='X')})
        )


def test_known_backends():
    assert known_backends() == ['cvista', 'vtk']


@pytest.mark.usefixtures('no_tools')
def test_dry_run_needs_checkout_without_shipped_manifest(tmp_path: Path):
    cfg = _config(tmp_path, backend='demo', url='https://example.invalid/demo.git')
    with pytest.raises(BuildError, match='--dry-run needs an existing checkout'):
        Builder(cfg, Console(quiet=True)).build()


@pytest.mark.usefixtures('no_tools')
def test_cloned_tree_without_manifest(tmp_path: Path, mocker: MockerFixture):
    cfg = _config(tmp_path, backend='demo', url='https://example.invalid/demo.git', dry_run=False)
    builder = Builder(cfg, Console(quiet=True))
    mocker.patch.object(
        builder, 'git', side_effect=lambda *_a, **_k: cfg.src_dir.mkdir(parents=True)
    )
    with pytest.raises(BuildError, match=r'No pyvista-build\.toml'):
        builder.phase_source()


def test_version_command_failure(tmp_path: Path, mocker: MockerFixture):
    src = tmp_path / 'src'
    (src / '.git').mkdir(parents=True)
    (src / 'pyvista-build.toml').write_text(
        'manifest_version = 1\n'
        '[backend]\nname = "demo"\nimport_name = "demo"\ndefault_ref = "v1"\n'
        'version_command = "{python} version.py"\n'
    )
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    cfg = _config(tmp_path, backend='demo', source=src, dry_run=False)
    builder = Builder(cfg, Console(quiet=True))
    builder.phase_source()
    mocker.patch.object(
        subprocess, 'run', side_effect=subprocess.CalledProcessError(3, ['x'], stderr='boom')
    )
    with pytest.raises(BuildError, match='version_command failed with exit status 3: boom'):
        builder.version_suffix()


def test_hook_expansion_keeps_windows_paths(tmp_path: Path, mocker: MockerFixture, demo):
    mocker.patch.object(phases.sys, 'executable', 'C:\\Python\\python.exe')
    builder = Builder(_config(tmp_path, os_name='windows'), Console(quiet=True))
    builder._manifest = demo
    parts = builder._expand('{python} prune.py {build_dir}')
    assert parts == ['C:\\Python\\python.exe', 'prune.py', str(builder.config.cmake_dir)]


def _builder(tmp_path: Path, mocker: MockerFixture, **kwargs) -> Builder:
    """Return a builder on the shipped VTK manifest with the source phase done."""
    cfg = _config(tmp_path, dry_run=False, **kwargs)
    builder = Builder(cfg, Console(record=True, width=4000, force_terminal=False))
    builder._manifest = find_manifest('vtk', None)
    builder.groups = resolve_groups(builder.manifest, cfg.os_name)
    builder.option_values = resolve_options(builder.manifest, None)
    mocker.patch.object(builder, 'version_suffix', return_value=None)
    return builder


def test_manifest_before_source(tmp_path: Path):
    with pytest.raises(BuildError, match='only available after the source phase'):
        _ = Builder(_config(tmp_path), Console(quiet=True)).manifest


def test_run_executes_and_reports(tmp_path: Path, mocker: MockerFixture):
    builder = _builder(tmp_path, mocker)
    cwd = tmp_path / 'made'
    builder.run(
        [sys.executable, '-c', 'import os; print(os.environ["DEMO"])'], cwd=cwd, env={'DEMO': 'x'}
    )
    assert cwd.is_dir()
    builder.run([sys.executable, '-c', 'raise SystemExit(3)'], fatal=False)
    with pytest.raises(BuildError, match='exit status 3'):
        builder.run([sys.executable, '-c', 'raise SystemExit(3)'])
    with pytest.raises(BuildError, match='Cannot run definitely-not-a-command'):
        builder.run(['definitely-not-a-command'])


def test_run_resolves_from_the_build_path(tmp_path: Path, mocker: MockerFixture):
    builder = _builder(tmp_path, mocker)
    assert builder.env['PATH'].startswith(tools.scripts_dir())
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    exe = bin_dir / 'demo-tool.exe'
    exe.write_text('')
    exe.chmod(0o755)
    builder.env['PATH'] = str(bin_dir)
    spy = mocker.patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0))
    builder.run(['demo-tool.exe', '--flag'])
    assert spy.call_args.args[0] == [str(exe), '--flag']


def test_run_logged_filters_noise(tmp_path: Path, mocker: MockerFixture):
    builder = _builder(tmp_path, mocker)
    log = tmp_path / 'build.log'
    script = tmp_path / 'noisy.py'
    script.write_text(
        'print("ok line"); print("x.cxx(1): warning C4244: conversion"); '
        'print("x.cxx(2): error C2065: undeclared"); raise SystemExit(1)'
    )
    with pytest.raises(BuildError):
        builder.run([sys.executable, str(script)], log=log)
    out = builder.console.export_text()
    assert 'ok line' in out
    assert 'error C2065: undeclared' in out
    assert 'C4244' not in out
    assert 'C4244' in log.read_text()
    assert 'full log is in' in out
    builder.run([sys.executable, '-c', 'print("clean")'], log=log)
    assert log.read_text() == 'clean\n'


def test_source_not_a_checkout(tmp_path: Path, mocker: MockerFixture):
    builder = _builder(tmp_path, mocker, source=tmp_path / 'plain')
    (tmp_path / 'plain').mkdir()
    mocker.patch.object(tools, 'detect', return_value=True)
    with pytest.raises(BuildError, match='not a git checkout'):
        builder.phase_source()


def test_existing_checkout_update_and_ref(tmp_path: Path, mocker: MockerFixture):
    src = tmp_path / 'src'
    (src / '.git').mkdir(parents=True)
    builder = _builder(tmp_path, mocker, source=src, update=True)
    git = mocker.patch.object(builder, 'git', return_value='')
    builder._existing_checkout(src)
    calls = [c.args for c in git.call_args_list]
    assert ('fetch', '--tags', 'origin') in calls
    assert any('checkout' in c and 'origin/v9.7.1' in c for c in calls)
    builder = _builder(tmp_path, mocker, source=src, ref='main')
    git = mocker.patch.object(
        builder, 'git', side_effect=lambda *a, **_: a[-1] if a[0] == 'rev-parse' else ''
    )
    builder._existing_checkout(src)
    assert any('checkout' in c and 'main' in c for c in [c.args for c in git.call_args_list])
    git = mocker.patch.object(builder, 'git', side_effect=subprocess.CalledProcessError(128, 'g'))
    with pytest.raises(BuildError, match='git status failed'):
        builder._existing_checkout(src)
    git = mocker.patch.object(
        builder, 'git', side_effect=lambda *a, **_: a[-1] if a[0] == 'rev-parse' else ''
    )
    mocker.patch.object(builder, 'confirm', return_value=False)
    builder._existing_checkout(src)
    assert not any('checkout' in c.args for c in git.call_args_list)


def test_prompts_print_brackets_verbatim():
    assert phases.Prompt('Start? [y/n, or a number of jobs]').prompt.plain.startswith(
        'Start? [y/n, or a number of jobs]'
    )
    assert phases.Confirm('Use [bold]?').prompt.plain.startswith('Use [bold]?')


def test_tools_pip_paths(tmp_path: Path, mocker: MockerFixture):
    mocker.patch.object(
        tools.shutil,
        'which',
        side_effect=lambda name, **_: None if name == 'ninja' else f'/usr/bin/{name}',
    )
    mocker.patch.object(tools, 'executable_version', return_value='99.0')
    mocker.patch.object(tools, 'python_build_packages', return_value=[])
    builder = _builder(tmp_path, mocker)
    mocker.patch.object(tools, 'pip_install_command', return_value=None)
    with pytest.raises(BuildError, match='no pip or uv found'):
        builder.phase_tools()
    mocker.patch.object(tools, 'pip_install_command', return_value=['pip', 'install', 'ninja'])
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=False)
    with pytest.raises(BuildError, match='Pass --yes'):
        builder.phase_tools()
    builder.config.yes = True
    mocker.patch.object(builder, 'run')
    with pytest.raises(BuildError, match='Still not usable after install: ninja'):
        builder.phase_tools()
    builder = _builder(tmp_path, mocker, install=True, yes=True)
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    mocker.patch.object(tools, 'pip_install_command', return_value=None)
    with pytest.raises(BuildError, match='--install needs pip or uv'):
        builder.phase_tools()


@pytest.mark.parametrize('repair', [True, False])
@pytest.mark.parametrize('install', [True, False])
@pytest.mark.parametrize('missing', [(), ('ninja', 'auditwheel')])
def test_install_list_follows_flags(
    tmp_path: Path, mocker: MockerFixture, repair, install, missing
):
    absent = set(missing)
    mocker.patch.object(
        tools.shutil,
        'which',
        side_effect=lambda name, **_: None if name in absent else f'/usr/bin/{name}',
    )
    mocker.patch.object(tools, 'executable_version', return_value='99.0')
    mocker.patch.object(tools, 'python_build_packages', return_value=[])
    mocker.patch.object(
        tools, 'pip_install_command', side_effect=lambda *s, **_: ['pip', 'install', *s]
    )
    builder = _builder(tmp_path, mocker, repair=repair, install=install, yes=True)
    run = mocker.patch.object(builder, 'run', side_effect=lambda *_a, **_k: absent.clear())
    builder.phase_tools()
    bullets = [
        line.removeprefix('  • ')
        for line in builder.console.export_text().splitlines()
        if line.startswith('  • ')
    ]
    specs = (['ninja>=1.11'] + (['auditwheel'] if repair else [])) if missing else []
    expected = [f'{spec} into {sys.prefix}' for spec in specs]
    if install:
        expected.append(f'the vtk wheel into {sys.prefix}, once it is built')
    assert bullets == expected
    assert run.called == bool(specs)
    if specs:
        assert run.call_args.args[0] == ['pip', 'install', *specs]


@pytest.mark.parametrize(
    ('kwargs', 'tty', 'answer', 'error'),
    [
        ({'yes': True}, False, None, None),
        ({'dry_run': True}, False, None, None),
        ({}, False, None, 'Pass --yes'),
        ({}, True, True, None),
        ({}, True, False, 'Nothing was installed'),
        ({'interactive': True}, True, True, None),
        ({'interactive': True, 'yes': True}, True, None, None),
    ],
)
def test_confirm_installs(tmp_path: Path, mocker: MockerFixture, kwargs, tty, answer, error):
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=tty)
    ask = mocker.patch.object(phases.Confirm, 'ask', return_value=answer)
    console = Console(record=True, width=4000, force_terminal=False)
    builder = Builder(_config(tmp_path, **{'dry_run': False, **kwargs}), console)
    builder.confirm_installs([])
    assert console.export_text() == ''
    if error:
        with pytest.raises(BuildError, match=error):
            builder.confirm_installs(['ninja into /env'])
    else:
        builder.confirm_installs(['ninja into /env'])
    assert ask.call_count == (answer is not None)
    assert console.export_text() == 'The build installs:\n  • ninja into /env\n'


def test_git_version_suffix(tmp_path: Path, mocker: MockerFixture):
    cfg = _config(tmp_path, dry_run=False)
    (cfg.src_dir / '.git').mkdir(parents=True)
    builder = Builder(cfg, Console(quiet=True))
    builder._manifest = find_manifest('vtk', None)
    mocker.patch.object(builder, 'git', return_value='v9.7.0')
    assert builder.version_suffix() is None
    err = subprocess.CalledProcessError(128, ['git'])

    def describe_fails(*args, **_):
        if 'describe' in args:
            raise err
        return 'abcdef12'

    mocker.patch.object(builder, 'git', side_effect=describe_fails)
    assert builder.version_suffix() == 'dev0+gabcdef12'
    mocker.patch.object(builder, 'git', side_effect=err)
    assert builder.version_suffix() is None


def test_configure_paths(tmp_path: Path, mocker: MockerFixture):
    builder = _builder(tmp_path, mocker, clean=True)
    cfg = builder.config
    cfg.cmake_dir.mkdir(parents=True)
    (cfg.cmake_dir / 'stale').write_text('')
    run = mocker.patch.object(builder, 'run')
    builder.phase_configure()
    assert not (cfg.cmake_dir / 'stale').exists()
    assert run.called
    assert (cfg.cmake_dir / phases.STATE_FILENAME).is_file()
    (cfg.cmake_dir / 'CMakeCache.txt').write_text('')
    builder = _builder(tmp_path, mocker)
    run = mocker.patch.object(builder, 'run')
    builder.phase_configure()
    assert not run.called
    assert 'already configured' in builder.console.export_text()
    builder = _builder(tmp_path, mocker, build_type='Debug')
    mocker.patch.object(
        builder, 'confirm', side_effect=lambda q, **_: not q.startswith('Reconfigure')
    )
    with pytest.raises(BuildError, match='keeps its earlier configuration'):
        builder.phase_configure()
    builder = _builder(tmp_path, mocker, reconfigure=True)
    mocker.patch.object(
        builder, 'confirm', side_effect=lambda q, **_: not q.startswith('Run cmake')
    )
    with pytest.raises(BuildError, match='Stopped before configure'):
        builder.phase_configure()


def test_configure_state(tmp_path: Path, mocker: MockerFixture):
    builder = _builder(tmp_path, mocker)
    cfg = builder.config
    run = mocker.patch.object(builder, 'run')
    builder.phase_configure()
    state_file = cfg.cmake_dir / phases.STATE_FILENAME
    stored = json.loads(state_file.read_text())
    assert stored == {'command': builder.configure_command(), 'env': {}}
    assert run.call_args.args[0] == stored['command']
    (cfg.cmake_dir / 'CMakeCache.txt').write_text('')
    mocker.patch.object(builder, 'version_suffix', return_value='dev0+gabcdef12')
    confirm = mocker.patch.object(builder, 'confirm', return_value=True)
    builder.phase_configure()
    assert 'new commit' in builder.console.export_text()
    assert [c.args[0] for c in confirm.call_args_list] == ['Run cmake?']
    assert '-DVTK_VERSION_SUFFIX=dev0+gabcdef12' in run.call_args.args[0]
    state_file.write_text('{')
    builder.phase_configure()
    assert 'differs from the one' in builder.console.export_text()
    assert confirm.call_args_list[-2].args[0] == 'Reconfigure?'
    state_file.write_text(json.dumps({'command': ['cmake'], 'env': {}}))
    confirm.reset_mock()
    builder.phase_configure()
    assert confirm.call_args_list[0].args[0] == 'Reconfigure?'


def test_clean_dry_run_keeps_tree(tmp_path: Path, mocker: MockerFixture):
    builder = _builder(tmp_path, mocker, clean=True)
    builder.config.dry_run = True
    builder.config.cmake_dir.mkdir(parents=True)
    (builder.config.cmake_dir / 'stale').write_text('')
    builder.phase_configure()
    assert (builder.config.cmake_dir / 'stale').exists()
    assert not (builder.config.cmake_dir / phases.STATE_FILENAME).exists()


def test_disabled_groups_unset_their_keys(tmp_path: Path, mocker: MockerFixture, demo):
    assert cmake_unset(demo, 'linux', groups=['a', 'core']) == ['B', 'C']
    assert cmake_unset(demo, 'macos', groups=['b', 'core']) == []
    builder = _builder(tmp_path, mocker)
    cmd = builder.configure_command()
    unset = ['-UVTK_DEFAULT_RENDER_WINDOW_HEADLESS', '-UVTK_OPENGL_HAS_EGL', '-UVTK_USE_X']
    unset.append('-UVTK_OPENGL_HAS_OSMESA')
    assert cmd[7:11] == unset
    builder.groups = [*builder.groups, 'headless-egl']
    cmd = builder.configure_command()
    assert '-UVTK_OPENGL_HAS_OSMESA' in cmd
    assert not any(a in cmd for a in unset[:3])


def test_build_wheel_repair_install_paths(tmp_path: Path, mocker: MockerFixture):
    builder = _builder(tmp_path, mocker, repair=True)
    mocker.patch.object(builder, 'confirm', return_value=False)
    with pytest.raises(BuildError, match='Stopped before building'):
        builder.phase_build()
    with pytest.raises(BuildError, match='Stopped before packaging'):
        builder.phase_wheel()
    builder.phase_repair()
    assert 'Skipped.' in builder.console.export_text()
    builder.phase_install()
    assert 'Not installed' in builder.console.export_text()
    with pytest.raises(BuildError, match='No wheel found'):
        builder._newest_wheel(tmp_path)

    builder = _builder(tmp_path, mocker, install=True, repair=True)
    run = mocker.patch.object(builder, 'run')
    cfg = builder.config
    cfg.dist_dir.mkdir(parents=True)
    (cfg.dist_dir / 'vtk-9.7.0-cp314-cp314-linux_x86_64.whl').write_bytes(b'')
    builder.phase_wheel()
    assert builder.wheel is not None
    assert builder.wheel.name.startswith('vtk-9.7.0')
    cfg.wheelhouse.mkdir(parents=True)
    (cfg.wheelhouse / 'vtk-9.7.0-cp314-cp314-manylinux_2_39_x86_64.whl').write_bytes(b'')
    builder.phase_repair()
    assert 'manylinux' in builder.wheel.name
    assert 'auditwheel' in run.call_args_list[-1].args[0]
    builder.phase_install()
    assert '--force-reinstall' in run.call_args.args[0]
    builder.summary()
    assert 'Wheel:' in builder.console.export_text()
    mocker.patch.object(tools, 'pip_install_command', return_value=None)
    builder.phase_install()
    assert 'install the wheel yourself' in builder.console.export_text()


def test_tool_helpers(mocker: MockerFixture):
    mocker.patch.object(tools.platform, 'system', return_value='Darwin')
    assert tools.host_os() == 'macos'
    mocker.patch.object(tools.platform, 'system', return_value='Windows')
    assert tools.host_os() == 'windows'
    mocker.patch.object(tools.platform, 'system', return_value='Linux')
    assert tools.host_os() == 'linux'
    assert tools.executable_version('/definitely/not/here') is None
    assert tools.executable_version(sys.executable) is not None
    mocker.patch.object(
        tools.subprocess,
        'run',
        return_value=subprocess.CompletedProcess([], 0, stdout='no digits'),
    )
    assert tools.executable_version('x') is None
    for os_name in OS_NAMES:
        assert tools.compiler_instructions(os_name)
    tool = tools.Tool('t', ('t',), spec='>=1')
    mocker.patch.object(tools.shutil, 'which', return_value='/usr/bin/t')
    mocker.patch.object(tools, 'executable_version', return_value=None)
    assert tools.detect(tool)
    mocker.patch.object(tools, 'executable_version', return_value='not.a.version')
    assert tools.detect(tool)


def test_option_coerce_and_schema_errors():
    flag = Option('f', help='', default=True, cmake='F={value}')
    assert flag.coerce('on') is True
    assert flag.coerce('off') is False
    with pytest.raises(ManifestError, match='expects a boolean'):
        flag.coerce('maybe')
    count = Option('n', help='', default=2, cmake='N={value}')
    with pytest.raises(ManifestError, match='expects an integer'):
        count.coerce('two')
    with pytest.raises(ManifestError, match="'name' must be a string"):
        parse_manifest(
            dict(MINIMAL_MANIFEST, backend={'name': 1, 'import_name': 'd', 'default_ref': 'v'})
        )
    with pytest.raises(ManifestError, match='must be a list of strings'):
        parse_manifest(dict(MINIMAL_MANIFEST, groups={'g': {'help': 'x', 'cmake': 'A=ON'}}))
    with pytest.raises(ManifestError, match=r'\[tools\]'):
        parse_manifest(dict(MINIMAL_MANIFEST, tools={'cmake': 3}))
    with pytest.raises(ManifestError, match="'cmake' must be a table, got an array"):
        parse_manifest(dict(MINIMAL_MANIFEST, cmake=[]))


@pytest.mark.usefixtures('no_tools')
def test_intro_explains_and_asks(tmp_path: Path, mocker: MockerFixture):
    console = Console(record=True, width=4000, force_terminal=False)
    Builder(_config(tmp_path, interactive=True), console).build()
    out = console.export_text()
    assert out.splitlines()[1] == 'The build runs these eight steps:'
    assert 'asks before running' not in out
    assert out.index('Proceed?') < out.index('1/8 Tools') < out.index('2/8 Source')
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    mocker.patch.object(phases.Confirm, 'ask', return_value=False)
    builder = Builder(_config(tmp_path, interactive=True, dry_run=False), console)
    with pytest.raises(BuildError, match='Stopped before the first step'):
        builder.build()
    assert 'asks before running any command; Enter takes the default' in console.export_text()


@pytest.mark.parametrize(
    ('kwargs', 'tty', 'answer', 'error'),
    [
        ({'yes': True}, False, None, None),
        ({'dry_run': True}, False, None, None),
        ({}, False, None, 'Pass --yes'),
        ({'interactive': True}, False, None, 'Pass --yes'),
        ({}, True, True, None),
        ({}, True, False, 'Stopped before the first step'),
        ({'yes': True, 'interactive': True}, True, None, None),
    ],
)
def test_confirm_start(tmp_path: Path, mocker: MockerFixture, kwargs, tty, answer, error):
    """Every run asks once before the first step unless ``--yes`` or ``--dry-run`` is given."""
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=tty)
    ask = mocker.patch.object(phases.Confirm, 'ask', return_value=answer)
    console = Console(record=True, width=4000, force_terminal=False)
    builder = Builder(_config(tmp_path, **{'dry_run': False, **kwargs}), console)
    source = mocker.patch.object(builder, 'phase_source', side_effect=RuntimeError('started'))
    with pytest.raises(BuildError if error else RuntimeError, match=error or 'started'):
        builder.build()
    assert ask.call_count == (answer is not None)
    assert source.called == (error is None)
    assert f'go in {tmp_path / "vtk-build"}' in console.export_text()


def test_confirm_asks_on_a_terminal(tmp_path: Path, mocker: MockerFixture):
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    ask = mocker.patch.object(phases.Confirm, 'ask', return_value=False)
    console = Console(record=True, width=4000, force_terminal=False)
    builder = Builder(_config(tmp_path, dry_run=False, interactive=True), console)
    assert builder.confirm('Go?') is False
    assert ask.call_args.args[0] == 'Go?'
    builder = Builder(_config(tmp_path, dry_run=False), console)
    assert builder.confirm('Go?') is True
    assert console.export_text() == ''
    builder = Builder(_config(tmp_path, interactive=True), console)
    assert builder.confirm('Go?') is True
    assert 'Go? -> yes (dry run)' in console.export_text()
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=False)
    builder = Builder(_config(tmp_path, interactive=True, dry_run=False), console)
    assert builder.confirm('Go [now]?', default=False) is False
    assert 'Go [now]? -> no (no terminal)' in console.export_text()


@pytest.mark.usefixtures('no_tools')
def test_clone_declined(tmp_path: Path, mocker: MockerFixture):
    builder = Builder(_config(tmp_path, dry_run=False), Console(quiet=True))
    mocker.patch.object(builder, 'confirm', return_value=False)
    with pytest.raises(BuildError, match='Nothing to build without a source checkout'):
        builder.phase_source()


def test_groups_phase_errors_and_required_display(tmp_path: Path, mocker: MockerFixture, demo):
    builder = _builder(tmp_path, mocker, with_=['nope'])
    with pytest.raises(BuildError, match='Unknown group'):
        builder.phase_groups()
    builder = _builder(tmp_path, mocker)
    builder._manifest = demo
    builder.groups = resolve_groups(demo, 'linux')
    builder.option_values = resolve_options(demo, None)
    builder._print_selection()
    assert '(required)' in builder.console.export_text()


def test_version_command_output(tmp_path: Path, mocker: MockerFixture):
    src = tmp_path / 'src'
    (src / '.git').mkdir(parents=True)
    (src / 'pyvista-build.toml').write_text(
        'manifest_version = 1\n'
        '[backend]\nname = "demo"\nimport_name = "demo"\ndefault_ref = "v1"\n'
        'version_command = "{python} version.py"\n'
    )
    builder = Builder(
        _config(tmp_path, backend='demo', source=src, dry_run=False), Console(quiet=True)
    )
    builder._manifest = find_manifest('demo', src)
    (src / 'version.py').write_text('print("post5")')
    assert builder.version_suffix() == 'post5'
    (src / 'version.py').write_text('')
    assert builder.version_suffix() is None
    mocker.patch.object(subprocess, 'run', side_effect=FileNotFoundError('gone'))
    with pytest.raises(BuildError, match='Cannot run version_command'):
        builder.version_suffix()


def test_newest_wheel_by_mtime(tmp_path: Path):
    old, new = tmp_path / 'b-old.whl', tmp_path / 'a-new.whl'
    old.write_bytes(b'')
    new.write_bytes(b'')
    os.utime(old, (1, 1))
    builder = Builder(_config(tmp_path), Console(quiet=True))
    assert builder._newest_wheel(tmp_path) == new
    assert builder._newest_wheel(tmp_path / 'empty') is None


@pytest.mark.parametrize('os_name', OS_NAMES)
def test_wheel_env_and_repair_command(tmp_path: Path, mocker: MockerFixture, os_name):
    mocker.patch.object(phases.platform, 'machine', return_value='arm64')
    builder = _builder(tmp_path, mocker, os_name=os_name)
    cfg = builder.config
    wheel = tmp_path / 'vtk.whl'
    cmd, env = builder.repair_command(wheel)
    out = str(cfg.wheelhouse)
    expected = {
        'linux': (['auditwheel', 'repair', str(wheel), '-w', out], {}),
        'windows': (
            [
                'delvewheel',
                'repair',
                '--add-path',
                str(cfg.cmake_dir / 'bin'),
                '-w',
                out,
                str(wheel),
            ],
            {},
        ),
        'macos': (
            ['delocate-wheel', '--require-archs', 'arm64', '-w', out, '-v', str(wheel)],
            {'DYLD_LIBRARY_PATH': str(cfg.cmake_dir / 'lib')},
        ),
    }
    assert (cmd, env) == expected[os_name]
    wheel_env = {'_PYTHON_HOST_PLATFORM': 'macosx-11.0-arm64', 'ARCHFLAGS': '-arch arm64'}
    assert builder.wheel_env() == (wheel_env if os_name == 'macos' else {})
    names = [t.name for t in tools.required_tools(os_name, {}, repair=True)]
    compiler = 'msvc' if os_name == 'windows' else 'c++ compiler'
    repairer = {'linux': 'auditwheel', 'macos': 'delocate-wheel', 'windows': 'delvewheel'}
    assert names == ['git', 'cmake', 'ninja', compiler, repairer[os_name]]
    assert repairer[os_name] not in [
        t.name for t in tools.required_tools(os_name, {}, repair=False)
    ]


def test_installer_and_build_packages(mocker: MockerFixture):
    mocker.patch.object(tools.importlib.util, 'find_spec', return_value=None)
    assert tools.python_build_packages() == ['setuptools', 'wheel']
    mocker.patch.object(tools.shutil, 'which', return_value='/opt/uv')
    assert tools.pip_install_command('x', args=('--no-deps',)) == [
        '/opt/uv',
        'pip',
        'install',
        '--python',
        sys.executable,
        '--no-deps',
        'x',
    ]
    mocker.patch.object(
        tools.importlib.util, 'find_spec', side_effect=lambda n: None if n == 'wheel' else object()
    )
    assert tools.python_build_packages() == ['wheel']


def test_build_asks_for_jobs(tmp_path: Path, mocker: MockerFixture):
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    mocker.patch.object(phases.os, 'cpu_count', return_value=8)
    mocker.patch.object(tools, 'physical_memory', return_value=64 * 2**30)
    builder = _builder(tmp_path, mocker)
    builder.config.interactive = True
    run = mocker.patch.object(builder, 'run')
    mocker.patch.object(builder, 'hooks')
    mocker.patch.object(phases.Prompt, 'ask', side_effect=['x', '0', '4'])
    builder.phase_build()
    out = builder.console.export_text()
    assert '8 CPU cores and 64 GB of memory detected; compiling with 8 jobs.' in out
    assert run.call_args_list[0].args[0][-2:] == ['--parallel', '4']
    mocker.patch.object(phases.Prompt, 'ask', return_value='y')
    builder.phase_build()
    assert run.call_args_list[-1].args[0][-2:] == ['--parallel', '8']
    mocker.patch.object(phases.Prompt, 'ask', return_value='n')
    with pytest.raises(BuildError, match='Stopped before building'):
        builder.phase_build()


def test_plan(tmp_path: Path):
    """The start summary says what every step does with the given settings."""
    cfg = _config(
        tmp_path,
        repair=True,
        install=True,
        jobs=3,
        defines=['X=1'],
        without=['web'],
        ref='v9.7.0',
    )
    plan = dict(Builder(cfg, Console(quiet=True)).plan())
    assert list(plan) == [
        'Tools',
        'Source',
        'Build options',
        'Configure',
        'Build',
        'Wheel',
        'Repair',
        'Install',
    ]
    assert (
        plan['Source']
        == f'clone https://gitlab.kitware.com/vtk/vtk.git at v9.7.0 into {cfg.src_dir}'
    )
    assert 'auditwheel' in plan['Tools']
    assert 'auditwheel' in plan['Repair']
    assert 'rendering' in plan['Build options']
    assert 'web' not in plan['Build options']
    assert plan['Configure'].endswith(', plus -DX=1')
    assert plan['Build'].startswith('compile with 3 parallel jobs')
    assert plan['Install'] == f'install the wheel into {sys.prefix}'
    with pytest.raises(BuildError, match='nope'):
        Builder(_config(tmp_path, with_=['nope']), Console(quiet=True)).plan()

    (cfg.src_dir / '.git').mkdir(parents=True)
    (cfg.src_dir / 'pyvista-build.toml').write_text('manifest_version = 99\n')
    plan = dict(
        Builder(
            _config(tmp_path, backend='demo', ref='main', repair=False), Console(quiet=True)
        ).plan()
    )
    assert plan['Source'] == (
        f'reuse the checkout at {cfg.src_dir}, local edits kept, checked out at main'
    )
    assert "checkout's manifest" in plan['Build options']
    assert plan['Repair'].startswith('skipped (--no-repair)')
    assert plan['Install'].startswith('skipped')


def _path_style(path: str) -> str:
    """Return the style a narrated POSIX path gets, unlinked where it has no file URI."""
    if os.name == 'nt':
        return 'magenta'
    return f'magenta link file://{path}'


def test_narration_highlighter_edge_cases(mocker: MockerFixture):
    """Paths stay one span, ``~`` alone is not a path, and an unlinkable path is still colored."""
    text = phases.Text('/a/2026-10/b about ~5 minutes')
    phases._HIGHLIGHT.highlight(text)
    spans = [(text.plain[s.start : s.end], str(s.style)) for s in text.spans]
    assert spans == [('/a/2026-10/b', _path_style('/a/2026-10/b')), ('5', 'cyan')]
    mocker.patch.object(
        phases,
        'Path',
        return_value=mocker.Mock(
            **{
                'expanduser.return_value.as_uri.side_effect': ValueError,
            }
        ),
    )
    text = phases.Text('flags /MP')
    phases._HIGHLIGHT.highlight(text)
    assert [str(s.style) for s in text.spans] == ['magenta']


def test_narration_highlighter():
    """URLs, paths, flags and versions get a style, other words do not."""
    text = phases.Text(
        'clone https://x.org/vtk.git at v9.7.1 into /a/b, --repair skipped, 15 jobs'
    )
    phases._HIGHLIGHT.highlight(text)
    styled = {text.plain[s.start : s.end]: str(s.style) for s in text.spans}
    assert styled.pop('/a/b') == _path_style('/a/b')
    assert styled == {
        'https://x.org/vtk.git': 'underline blue',
        '--repair': 'bold cyan',
        'skipped': 'yellow',
        'v9.7.1': 'cyan',
        '15': 'cyan',
    }


def test_update_follows_the_remote_branch(tmp_path: Path, mocker: MockerFixture):
    """``--update`` on a branch ref checks out the fetched commit, not the stale local branch."""

    def git(*args, cwd):
        return subprocess.run(
            [str(shutil.which('git')), '-c', 'user.name=t', '-c', 'user.email=t@t', *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    origin = tmp_path / 'origin'
    origin.mkdir()
    git('init', '-q', '-b', 'main', cwd=origin)
    git('commit', '-q', '--allow-empty', '-m', 'one', cwd=origin)
    git('clone', '-q', str(origin), str(tmp_path / 'src'), cwd=tmp_path)
    git('commit', '-q', '--allow-empty', '-m', 'two', cwd=origin)
    builder = _builder(tmp_path, mocker, source=tmp_path / 'src', update=True, ref='main')
    builder._existing_checkout(tmp_path / 'src')
    assert git('rev-parse', 'HEAD', cwd=tmp_path / 'src') == git('rev-parse', 'HEAD', cwd=origin)
    (tmp_path / 'src' / 'notes.txt').write_text('mine')
    git('tag', 'v1', cwd=origin)
    builder.config.ref = 'v1'
    builder._existing_checkout(tmp_path / 'src')
    assert git('describe', '--tags', cwd=tmp_path / 'src') == 'v1'


def test_tools_table_is_validated():
    """Unknown tools and malformed version specifiers are manifest errors."""
    with pytest.raises(ManifestError, match=r'\[tools\]'):
        parse_manifest(dict(MINIMAL_MANIFEST, tools={'cmak': '>=3'}))
    with pytest.raises(ManifestError, match='not a version specifier'):
        parse_manifest(dict(MINIMAL_MANIFEST, tools={'cmake': '3.22'}))


def test_plan_rejects_unknown_backend(tmp_path: Path):
    """The start summary fails on a backend with no URL before asking to proceed."""
    builder = Builder(_config(tmp_path, backend='nosuch'), Console(quiet=True))
    with pytest.raises(BuildError, match='Unknown backend'):
        builder.plan()
    with pytest.raises(BuildError, match='Unknown backend'):
        builder.phase_source()


def test_plan_wording_under_yes_and_source(tmp_path: Path, mocker: MockerFixture):
    """``--yes`` and ``--source`` change what the summary promises."""
    console = Console(record=True, width=4000, force_terminal=False)
    builder = Builder(_config(tmp_path, yes=True, source=tmp_path / 'elsewhere'), console)
    mocker.patch.object(builder, 'phase_source', side_effect=RuntimeError('started'))
    with pytest.raises(RuntimeError):
        builder.build()
    out = console.export_text()
    assert 'without asking (--yes)' in out
    assert 'Build tree and wheel go in' in out


def test_wheel_env_takes_the_define(tmp_path: Path, mocker: MockerFixture):
    """A ``-D CMAKE_OSX_DEPLOYMENT_TARGET`` sets the wheel's platform tag."""
    builder = _builder(
        tmp_path, mocker, os_name='macos', defines=['CMAKE_OSX_DEPLOYMENT_TARGET:STRING=12.0']
    )
    assert builder.wheel_env()['_PYTHON_HOST_PLATFORM'].startswith('macosx-12.0-')


def test_narration_escapes_paths(tmp_path: Path):
    """Brackets in a path are printed, not read as markup."""
    console = Console(record=True, width=4000, force_terminal=False)
    cfg = _config(tmp_path, build_dir=tmp_path / 'b[x]')
    Builder(cfg, console).summary()
    assert 'b[x]' in console.export_text()


@pytest.fixture(autouse=True)
def _unknown_memory(mocker: MockerFixture):
    """Make the default job count one per core, whatever memory the test machine has."""
    mocker.patch.object(tools, 'physical_memory', return_value=None)


def test_default_jobs_follow_memory(tmp_path: Path, mocker: MockerFixture):
    """The default job count is one per core, at most one per 2 GB of memory."""
    gib = 2**30
    assert tools.default_jobs(8, None) == 8
    assert tools.default_jobs(0, None) == 1
    assert tools.default_jobs(8, 64 * gib) == 8
    assert tools.default_jobs(16, 24 * gib) == 12
    assert tools.default_jobs(8, gib) == 1
    mocker.patch.object(phases.os, 'cpu_count', return_value=16)
    mocker.patch.object(tools, 'physical_memory', return_value=8 * gib)
    builder = _builder(tmp_path, mocker)
    mocker.patch.object(builder, 'run')
    mocker.patch.object(builder, 'hooks')
    builder.config.yes = True
    builder.phase_build()
    out = builder.console.export_text()
    assert '16 CPU cores and 8 GB of memory detected; compiling with 4 jobs, one per 2 GB' in out
    assert dict(builder.plan())['Build'].startswith(
        'compile with 4 parallel jobs (one per 2 GB of memory)'
    )
    builder.config.jobs = 6
    assert dict(builder.plan())['Build'].startswith('compile with 6 parallel jobs, logging')


def test_physical_memory(mocker: MockerFixture):
    """Physical memory comes from sysconf, or GlobalMemoryStatusEx on Windows."""
    mocker.stopall()
    assert tools.physical_memory() > 0
    mocker.patch.object(tools.os, 'sysconf', side_effect=ValueError, create=True)
    mocker.patch.object(tools.sys, 'platform', 'linux')
    assert tools.physical_memory() is None
    import ctypes

    def status(ref):
        ref._obj.ullTotalPhys = 16 * 2**30
        return 1

    kernel32 = mocker.Mock(**{'GlobalMemoryStatusEx.side_effect': status})
    mocker.patch.object(ctypes, 'windll', mocker.Mock(kernel32=kernel32), create=True)
    mocker.patch.object(tools.sys, 'platform', 'win32')
    assert tools.physical_memory() == 16 * 2**30
    kernel32.GlobalMemoryStatusEx.side_effect = None
    kernel32.GlobalMemoryStatusEx.return_value = 0
    assert tools.physical_memory() is None


def test_recheck_tools_after_clone(tmp_path: Path, mocker: MockerFixture):
    """Tool versions only the cloned manifest asks for are checked after the clone."""
    builder = _builder(tmp_path, mocker)
    shipped = builder.manifest
    run = mocker.patch.object(builder, 'run')
    confirm = mocker.patch.object(builder, 'confirm_installs')
    builder.recheck_tools()
    assert not run.called
    builder._checked_tools = dict(shipped.tools)
    builder.recheck_tools()
    assert not run.called

    builder._manifest = replace(shipped, tools={**shipped.tools, 'cmake': '>=99'})
    detect = mocker.patch.object(tools, 'detect', return_value=True)
    builder.recheck_tools()
    assert "The checkout's manifest asks for cmake>=99" in builder.console.export_text()
    assert not run.called
    assert builder._checked_tools['cmake'] == '>=99'

    detect.reset_mock()
    builder._checked_tools = {}
    builder._manifest = replace(shipped, tools={'cmake': '', 'ninja': ''})
    builder.recheck_tools()
    assert not detect.called

    builder._checked_tools = {}
    builder._manifest = replace(shipped, tools={'cmake': '>=99'})
    detect.return_value = False
    mocker.patch.object(tools, 'pip_install_command', return_value=None)
    with pytest.raises(BuildError, match='no pip or uv found'):
        builder.recheck_tools()
    builder._checked_tools = {}
    mocker.patch.object(tools, 'pip_install_command', return_value=['pip', 'install'])
    with pytest.raises(BuildError, match='Still not usable after install: cmake>=99'):
        builder.recheck_tools()
    assert confirm.call_args.args[0] == [f'cmake>=99 into {sys.prefix}']
    builder._checked_tools = {}
    detect.side_effect = [False, True]
    builder.recheck_tools()
    assert run.call_args.args[0] == ['pip', 'install']
    builder._checked_tools = {}
    builder.config.dry_run = True
    detect.side_effect = None
    builder.recheck_tools()
    assert run.call_count == 3
