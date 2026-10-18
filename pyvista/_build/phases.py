"""The build itself: source, tools, groups, configure, build, wheel, repair, install."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

from rich.console import Console
from rich.highlighter import Highlighter
from rich.markup import escape
from rich.prompt import Confirm as _RichConfirm
from rich.prompt import Prompt as _RichPrompt
from rich.style import Style
from rich.table import Table
from rich.text import Text

from . import manylinux as _manylinux
from . import tools as _tools
from .manifest import MANIFEST_FILENAME
from .manifest import ManifestError
from .manifest import cmake_entries
from .manifest import cmake_unset
from .manifest import option_environment
from .manifest import resolve_groups
from .manifest import resolve_options
from .registry import BACKENDS
from .registry import find_manifest
from .registry import known_backends
from .registry import shipped_manifest

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from .manifest import Manifest

STATE_FILENAME = '.pyvista-build.json'
BUILD_TYPES = ('Release', 'RelWithDebInfo', 'Debug')
COMPILER_CACHES = ('auto', 'ccache', 'sccache', 'none')


class _NarrationHighlighter(Highlighter):
    """Color URLs, paths, command-line flags and version numbers."""

    rules = (
        (re.compile(r'(https?://|git@)\S*[^\s.,)]'), 'underline blue'),
        (re.compile(r'(?<![\w-])--?[a-zA-Z][\w-]*'), 'bold cyan'),
        (re.compile(r'(?<![\w.])v?\d+(\.\d+)*(?!\w)'), 'cyan'),
        (re.compile(r'\b[Ss]kipped\b'), 'yellow'),
    )
    path = re.compile(r'(?<![\w/.:])(~(?=/)|/)[-\w.+~/]*[-\w+~/]')

    def highlight(self, text: Text) -> None:
        """Apply every rule to the text in place, leaving paths whole.

        Parameters
        ----------
        text : Text
            Text to color.

        """
        paths = []
        for match in self.path.finditer(text.plain):
            try:
                link = Path(match.group()).expanduser().as_uri()
            except (ValueError, RuntimeError):
                link = None
            text.stylize(Style(color='magenta', link=link), *match.span())
            paths.append(match.span())
        for pattern, style in self.rules:
            for match in pattern.finditer(text.plain):
                start, end = match.span()
                if not any(a <= start < b for a, b in paths):
                    text.stylize(style, start, end)


_HIGHLIGHT = _NarrationHighlighter()
_TAG_LIKE = re.compile(r'v?\d+\.\d+')
_join = subprocess.list2cmdline if os.name == 'nt' else shlex.join
_NOISY_BUILD_LINE = re.compile(
    r': warning C\d+:|: note:|^\s+(with|\[|\])\s*$|^\s+[A-Za-z_][A-Za-z0-9_]*='
)


class BuildError(Exception):
    """A step cannot continue; the message tells the user what to do."""


class Prompt(_RichPrompt):
    """Text prompt whose question is printed verbatim.

    Parameters
    ----------
    prompt : str, default: ''
        Question text, shown as is.

    **kwargs
        Passed to the rich prompt.

    """

    def __init__(self, prompt: str = '', **kwargs) -> None:
        super().__init__(Text(prompt, style='prompt'), **kwargs)


class Confirm(_RichConfirm):
    """Yes/no prompt whose question is printed verbatim.

    Parameters
    ----------
    prompt : str, default: ''
        Question text, shown as is.

    **kwargs
        Passed to the rich prompt.

    """

    def __init__(self, prompt: str = '', **kwargs) -> None:
        super().__init__(Text(prompt, style='prompt'), **kwargs)


@dataclass
class BuildConfig:
    """Everything the command line decided."""

    backend: str
    build_dir: Path
    source: Path | None = None
    url: str | None = None
    ref: str | None = None
    with_: list[str] = field(default_factory=list)
    without: list[str] = field(default_factory=list)
    all_: bool = False
    minimal: bool = False
    build_type: str = 'Release'
    jobs: int | None = None
    defines: list[str] = field(default_factory=list)
    options: list[str] = field(default_factory=list)
    clean: bool = False
    reconfigure: bool = False
    update: bool = False
    compiler_cache: str = 'none'
    cache_dir: Path | None = None
    version_suffix: str | None = None
    repair: bool = True
    install: bool = False
    yes: bool = False
    interactive: bool = False
    dry_run: bool = False
    manylinux: str | None = None
    os_name: str = field(default_factory=_tools.host_os)

    @property
    def src_dir(self) -> Path:  # numpydoc ignore=RT01
        """Source checkout: ``--source`` when given, else ``<build-dir>/src``."""
        return self.source if self.source is not None else self.build_dir / 'src'

    @property
    def cmake_dir(self) -> Path:  # numpydoc ignore=RT01
        """CMake build tree."""
        return self.build_dir / 'build'

    @property
    def dist_dir(self) -> Path:  # numpydoc ignore=RT01
        """Where ``bdist_wheel`` writes the raw wheel."""
        return self.cmake_dir / 'dist'

    @property
    def wheelhouse(self) -> Path:  # numpydoc ignore=RT01
        """Where the repaired wheel goes."""
        return self.build_dir / 'wheelhouse'

    @property
    def compiler_cache_dir(self) -> Path:  # numpydoc ignore=RT01
        """Compiler cache directory."""
        return self.cache_dir if self.cache_dir is not None else self.build_dir / 'compiler-cache'

    @property
    def prompting(self) -> bool:  # numpydoc ignore=RT01
        """Whether to ask questions: interactive, on a terminal, and not a dry run."""
        return self.interactive and not self.dry_run and sys.stdin.isatty()


class Builder:
    """Run the phases in order.

    Parameters
    ----------
    config : BuildConfig
        What to build and how.

    console : rich.console.Console, optional
        Console to print to.

    """

    def __init__(self, config: BuildConfig, console: Console | None = None) -> None:
        self.config = config
        self.console = console or Console()
        self.env: dict[str, str] = dict(os.environ)
        self.env['PATH'] = f'{_tools.scripts_dir()}{os.pathsep}{self.env.get("PATH", "")}'
        self._manifest: Manifest | None = None
        self.groups: list[str] = []
        self.option_values: dict[str, bool | int | str] = {}
        self.launcher: _tools.Tool | None = None
        self.wheel: Path | None = None

    @property
    def manifest(self) -> Manifest:  # numpydoc ignore=RT01
        """Manifest of the backend being built."""
        if self._manifest is None:
            msg = 'The manifest is only available after the source phase.'
            raise BuildError(msg)
        return self._manifest

    def step(self, title: str) -> None:
        """Print a phase heading.

        Parameters
        ----------
        title : str
            Heading text.

        """
        self.console.rule(f'[bold]{title}')

    def say(self, message: str) -> None:
        """Print a line of narration.

        Parameters
        ----------
        message : str
            Rich markup to print.

        """
        self.console.print(_HIGHLIGHT(Text.from_markup(message)))

    def confirm(self, question: str, *, default: bool = True) -> bool:
        """Ask a yes/no question, or take the default when not interactive.

        Parameters
        ----------
        question : str
            Question to ask.

        default : bool, default: True
            Answer taken on Enter, and without ``--interactive`` or under ``--dry-run``.

        Returns
        -------
        bool
            The answer.

        """
        if self.config.prompting:
            return Confirm.ask(question, default=default, console=self.console)
        if self.config.interactive:
            why = 'dry run' if self.config.dry_run else 'no terminal'
            answer = 'yes' if default else 'no'
            self.say(f'[dim]{escape(question)} -> {answer} ({why})[/dim]')
        return default

    def confirm_installs(self, items: Sequence[str]) -> None:
        """List everything the build will install and ask once; ``--yes`` answers yes.

        Parameters
        ----------
        items : Sequence[str]
            One line per thing to install, saying what and where.

        """
        if not items:
            return
        self.say('The build installs:')
        for item in items:
            self.say(f'  • {escape(item)}')
        if self.config.yes or self.config.dry_run:
            return
        if not sys.stdin.isatty():
            msg = 'Pass --yes to install these, or install them yourself and run again.'
            raise BuildError(msg)
        if not Confirm.ask('Install them?', default=True, console=self.console):
            msg = 'Nothing was installed; install them yourself and run again.'
            raise BuildError(msg)

    def _planned_manifest(self) -> Manifest | None:
        """Return the manifest the build will most likely use, or ``None`` before a clone."""
        cfg = self.config
        try:
            if (cfg.src_dir / '.git').exists():
                return find_manifest(cfg.backend, cfg.src_dir)
            if cfg.url is None and shipped_manifest(cfg.backend) is not None:
                return find_manifest(cfg.backend, None)
        except ManifestError:
            pass
        return None

    def plan(self) -> list[tuple[str, str]]:
        """Describe what each step will do, before any of them runs.

        Returns
        -------
        list[tuple[str, str]]
            Step name and description, in order.

        """
        cfg = self.config
        manifest = self._planned_manifest()
        if (cfg.src_dir / '.git').exists():
            source = f'reuse the checkout at {cfg.src_dir}, local edits kept'
            if cfg.ref:
                source += f', checked out at {cfg.ref}'
        else:
            url = cfg.url or BACKENDS.get(cfg.backend) or (manifest.url if manifest else None)
            ref = cfg.ref or (manifest.default_ref if manifest else None) or 'the default branch'
            source = f'clone {url} at {ref} into {cfg.src_dir}'
        tools = _tools.required_tools(
            cfg.os_name, manifest.tools if manifest else {}, repair=cfg.repair
        )
        names = ', '.join([t.label for t in tools] + list(_tools.PYTHON_BUILD_PACKAGES))
        tools_text = (
            f'check {names}. Anything missing is listed and installed into {sys.prefix} '
            'only after one more confirmation'
        )
        if manifest is None:
            options_text = "the default module groups and options from the checkout's manifest"
        else:
            try:
                groups = resolve_groups(
                    manifest,
                    cfg.os_name,
                    with_=cfg.with_,
                    without=cfg.without,
                    all_=cfg.all_,
                    minimal=cfg.minimal,
                )
                values = resolve_options(manifest, cfg.options)
            except ManifestError as err:
                raise BuildError(str(err)) from None
            options_text = f'module groups {", ".join(groups) or "(none)"}'
            if values:
                options_text += '. Options ' + ', '.join(f'{k}={v}' for k, v in values.items())
        configure = f'cmake, {cfg.build_type} build with Ninja, into {cfg.cmake_dir}'
        if cfg.defines:
            configure += ', plus ' + ' '.join(f'-D{d}' for d in cfg.defines)
        cache = {'none': 'off', 'auto': 'ccache or sccache, whichever is on PATH'}
        configure += f'. Compiler cache: {cache.get(cfg.compiler_cache, cfg.compiler_cache)}'
        jobs = cfg.jobs or os.cpu_count() or 1
        build = f'compile with {jobs} parallel jobs, logging to {cfg.build_dir / "build.log"}'
        wheel = f'package the wheel with setup.py bdist_wheel into {cfg.dist_dir}'
        if cfg.repair:
            repairer = _tools.required_tools(cfg.os_name, {}, repair=True)[-1].name
            repair = (
                f'bundle the libraries the wheel links against with {repairer}, '
                f'into {cfg.wheelhouse}, so it installs on other machines. '
                '--no-repair skips it'
            )
        else:
            repair = 'skipped (--no-repair), so the wheel may only work on this machine'
        if cfg.install:
            install = f'install the wheel into {sys.prefix}'
        else:
            install = 'skipped. The install command is printed at the end, or pass --install'
        return [
            ('Source', source),
            ('Tools', tools_text),
            ('Build options', options_text),
            ('Configure', configure),
            ('Build', build),
            ('Wheel', wheel),
            ('Repair', repair),
            ('Install', install),
        ]

    def confirm_start(self) -> None:
        """Say where the build goes and ask before the first step; ``--yes`` answers yes."""
        cfg = self.config
        self.say(
            f'Source, build tree and wheel go in [bold]{escape(str(cfg.build_dir))}[/bold]. '
            'Compiling takes from minutes to over an hour and needs several GB of disk.'
        )
        if (cfg.yes or cfg.dry_run) and not cfg.prompting:
            self.confirm('Proceed? (n stops)')
            return
        if not sys.stdin.isatty():
            msg = 'Pass --yes to build without a terminal.'
            raise BuildError(msg)
        if not Confirm.ask('Proceed? (n stops)', default=True, console=self.console):
            msg = 'Stopped before the first step.'
            raise BuildError(msg)

    def run(
        self,
        cmd: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        log: Path | None = None,
        fatal: bool = True,
    ) -> None:
        """Print and run a command; raise :class:`BuildError` when it fails.

        Parameters
        ----------
        cmd : Sequence[str]
            Command and arguments.

        cwd : Path, optional
            Working directory, created when missing.

        env : Mapping[str, str], optional
            Environment entries added to the builder's environment.

        log : Path, optional
            File that receives the full output; the console gets a filtered view.

        fatal : bool, default: True
            Raise on a non-zero exit status instead of only reporting it.

        """
        where = f' (in {cwd})' if cwd is not None else ''
        prefix = ''.join(f'{k}={shlex.quote(v)} ' for k, v in (env or {}).items())
        self.say(f'[cyan]$ {escape(prefix + _join(cmd))}[/cyan]{escape(where)}')
        if self.config.dry_run:
            return
        if cwd is not None:
            cwd.mkdir(parents=True, exist_ok=True)
        full_env = dict(self.env)
        full_env.update(env or {})
        resolved = [_tools.which(cmd[0], full_env), *cmd[1:]]
        try:
            if log is None:
                status = subprocess.run(resolved, cwd=cwd, env=full_env, check=False).returncode
            else:
                status = self._run_logged(resolved, cwd=cwd, env=full_env, log=log)
        except OSError as err:
            msg = f'Cannot run {cmd[0]}: {err}'
            raise BuildError(msg) from None
        if status != 0 and fatal:
            msg = f'Command failed with exit status {status}: {_join(cmd)}'
            raise BuildError(msg)

    def _run_logged(
        self, cmd: Sequence[str], *, cwd: Path | None, env: dict[str, str], log: Path
    ) -> int:
        """Stream a command's output with compiler noise filtered; keep the full log."""
        with (
            log.open('w', encoding='utf-8') as log_file,
            subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors='replace',
            ) as proc,
        ):
            for line in proc.stdout or ():
                log_file.write(line)
                if not _NOISY_BUILD_LINE.search(line):
                    self.console.out(line, end='', highlight=False)
        if proc.returncode != 0:
            self.say(f'[red]Build failed; the full log is in {log}[/red]')
        return proc.returncode

    def git(self, *args: str, cwd: Path | None = None, capture: bool = False) -> str:
        """Run git.

        Parameters
        ----------
        *args : str
            Arguments after ``git``.

        cwd : Path, optional
            Repository to run in.

        capture : bool, default: False
            Return the output instead of streaming it; errors then raise
            :class:`subprocess.CalledProcessError`.

        Returns
        -------
        str
            Stripped output when ``capture`` is set, else an empty string.

        """
        cmd = ['git', *args]
        if capture:
            return subprocess.run(
                cmd, cwd=cwd, env=self.env, capture_output=True, text=True, check=True
            ).stdout.strip()
        self.run(cmd, cwd=cwd)
        return ''

    @property
    def python(self) -> str:  # numpydoc ignore=RT01
        """Interpreter the wheel is built for."""
        return sys.executable

    def hooks(self, name: str) -> None:
        """Run the manifest's hook commands inside the source tree, with the options' variables.

        Parameters
        ----------
        name : str
            Hook name, for example ``'post_build'``.

        """
        cfg = self.config
        for template in self.manifest.hooks.get(name, ()):
            self.run(self._expand(template), cwd=cfg.src_dir, env=self.configure_env())

    def _expand(self, template: str) -> list[str]:
        """Split a manifest command template, then substitute the placeholders per part."""
        cfg = self.config
        fields = {
            'python': self.python,
            'src_dir': cfg.src_dir,
            'build_dir': cfg.cmake_dir,
            'dist_dir': cfg.dist_dir,
            'wheelhouse': cfg.wheelhouse,
        }
        return [part.format(**fields) for part in shlex.split(template)]

    def build(self) -> Path | None:
        """Run every phase.

        Returns
        -------
        Path | None
            The final wheel, or ``None`` on a dry run.

        """
        self.console.print(
            f'[bold green]Building backend: {escape(self.config.backend)}[/bold green]'
        )
        if self.config.manylinux:
            return self.build_in_manylinux()
        if self.config.interactive:
            self.say(
                'Eight steps: source, tools, build options, configure, build, wheel, repair, '
                'and install.'
            )
        else:
            self.say('The build runs these eight steps:')
            for number, (name, text) in enumerate(self.plan(), start=1):
                self.say(f'  {number}. [bold]{name}[/bold]: {escape(text)}')
        if self.config.prompting:
            self.say(
                'Each step explains what it will do and asks before running any command. Enter '
                'takes the default shown in parentheses, and each question says what n and any '
                'other answer do. Ctrl-C stops at any time.'
            )
        self.confirm_start()
        self.phase_source()
        self.phase_tools()
        self.phase_groups()
        self.phase_configure()
        self.phase_build()
        self.phase_wheel()
        self.phase_repair()
        self.phase_install()
        self.summary()
        return self.wheel

    def phase_source(self) -> None:
        """Clone or reuse the checkout, then load its manifest."""
        cfg = self.config
        self.step('1/8 Source')
        git = _tools.git_tool()
        if not _tools.detect(git, self.env):
            msg = f'git was not found on PATH. {git.instructions}'
            raise BuildError(msg)

        shipped = shipped_manifest(cfg.backend)
        try:
            self._source_phase(shipped)
        except ManifestError as err:
            raise BuildError(str(err)) from None

    def _source_phase(self, shipped: Path | None) -> None:
        """Body of :meth:`phase_source`; manifest errors propagate to the caller."""
        cfg = self.config
        url = cfg.url or BACKENDS.get(cfg.backend)
        if url is None and shipped is not None:
            url = find_manifest(cfg.backend, None).url
        replacing = False
        if cfg.source is not None:
            if not (cfg.source / '.git').exists():
                msg = f'--source {cfg.source} is not a git checkout.'
                raise BuildError(msg)
            self.say(f'Using the existing checkout at [bold]{cfg.source}[/bold].')
            self._existing_checkout(cfg.source)
        elif (cfg.src_dir / '.git').exists() and not (replacing := self._replace_checkout()):
            self.say(f'Reusing the checkout at [bold]{cfg.src_dir}[/bold]; local edits are kept.')
            self._existing_checkout(cfg.src_dir)
        else:
            if url is None:
                msg = (
                    f'Unknown backend {cfg.backend!r}. Known backends: {known_backends()}. '
                    'Pass --url <git url> or --source <checkout> to build another one.'
                )
                raise BuildError(msg)
            ref = cfg.ref
            if ref is None and shipped is not None:
                ref = find_manifest(cfg.backend, None).default_ref
            if not replacing and cfg.src_dir.is_dir() and any(cfg.src_dir.iterdir()):
                msg = (
                    f'{cfg.src_dir} exists but is not a git checkout. Move it away, or pass '
                    '--build-dir to build somewhere else.'
                )
                raise BuildError(msg)
            if not replacing:
                self.say(
                    f'The source will be cloned from [bold]{url}[/bold] into '
                    f'[bold]{cfg.src_dir}[/bold] (blob-less partial clone: history without '
                    'file contents until needed).'
                )
                if cfg.prompting:
                    self.say(
                        'To build a fork instead, answer with its git URL (https://... or '
                        'git@...), or with the path of a local checkout to build that one.'
                    )
                source, _ = self._ask_source(
                    'Clone it now?', stop='Nothing to build without a source checkout.'
                )
                if source is not None:
                    self.say(f'Using the existing checkout at [bold]{escape(str(source))}[/bold].')
                    cfg.source = source
                    self._existing_checkout(source)
                    self._load_manifest(shipped)
                    return
            url, ref = cfg.url or url, cfg.ref or ref
            if replacing and not cfg.dry_run:
                shutil.rmtree(cfg.src_dir)
            self.git('clone', '--filter=blob:none', url, str(cfg.src_dir))
            if ref is None and not cfg.dry_run:
                ref = find_manifest(cfg.backend, cfg.src_dir).default_ref
            if ref is not None:
                self.git('-c', 'advice.detachedHead=false', 'checkout', ref, cwd=cfg.src_dir)

        self._load_manifest(shipped)

    def _load_manifest(self, shipped: Path | None) -> None:
        """Load the manifest from the checkout, falling back to the shipped one."""
        cfg = self.config
        if cfg.dry_run and not (cfg.src_dir / MANIFEST_FILENAME).is_file() and shipped is None:
            msg = (
                f'--dry-run needs an existing checkout for {cfg.backend!r}: its '
                f'{MANIFEST_FILENAME} lives in the source tree.'
            )
            raise BuildError(msg)
        source = cfg.src_dir if cfg.src_dir.exists() else None
        self._manifest = find_manifest(cfg.backend, source)
        self.say(f'Manifest: [bold]{self.manifest.path}[/bold]')

    def _replace_checkout(self) -> bool:
        """Ask whether to keep the checkout in the build tree or replace it.

        Returns
        -------
        bool
            ``True`` when a fork is to be cloned in its place.

        """
        cfg = self.config
        if not cfg.prompting:
            if cfg.url is not None:
                self.say(
                    f'[yellow]--url is not used: the checkout at {cfg.src_dir} is reused. '
                    'Pass --build-dir to clone the URL somewhere else.[/yellow]'
                )
            return False
        self.say(
            f'A checkout already exists at [bold]{cfg.src_dir}[/bold]. Answer y to build it '
            'as is, a git URL (https://... or git@...) to delete it and clone that fork in '
            'its place, or the path of a local checkout to build that one.'
        )
        source, typed_url = self._ask_source(
            'Reuse it?',
            stop=f'Stopped. Delete {cfg.src_dir} or pass --build-dir to build elsewhere.',
        )
        if source is not None:
            cfg.source = source
            return False
        if not typed_url:
            return False
        self.say(f'[yellow]{cfg.src_dir} will be deleted and cloned again.[/yellow]')
        if not self.confirm('Delete it? (n stops and keeps it)', default=False):
            msg = 'Keeping the existing checkout. Run again to build it, or pass --build-dir.'
            raise BuildError(msg)
        return True

    def _ask_source(self, question: str, *, stop: str) -> tuple[Path | None, bool]:
        """Ask ``question``, taking a git URL or a local checkout instead of a yes.

        Parameters
        ----------
        question : str
            Question to ask. Yes keeps the current plan.

        stop : str
            Error raised when the answer is no.

        Returns
        -------
        Path or None
            An existing checkout to use instead, else ``None``.

        bool
            Whether a git URL was typed; it is then in ``config.url``.

        """
        cfg = self.config
        if not cfg.prompting:
            if not self.confirm(question):
                raise BuildError(stop)
            return None, False
        while True:
            answer = Prompt.ask(
                f'{question} [y / n = stop / git URL / path to a checkout]',
                default='y',
                console=self.console,
            ).strip()
            if answer.lower() in ('y', 'yes'):
                return None, False
            if answer.lower() in ('n', 'no'):
                raise BuildError(stop)
            path = Path(answer).expanduser()
            if (path / '.git').exists():
                return path.resolve(), False
            if path.exists() or ('://' not in answer and not answer.startswith('git@')):
                self.say(f'[red]{escape(answer)} is neither a git checkout nor a git URL.[/red]')
                continue
            cfg.url = answer
            cfg.ref = Prompt.ask(
                'Tag, branch or commit to check out (e.g. v9.7.0, master, or a SHA)',
                default=cfg.ref or self._default_ref(),
                console=self.console,
            )
            self.say(
                f'Cloning [bold]{escape(cfg.url)}[/bold] at [bold]{escape(cfg.ref)}[/bold] '
                'instead.'
            )
            return None, True

    def _default_ref(self) -> str:
        """Return the shipped manifest's default ref, or ``'master'`` without one."""
        if shipped_manifest(self.config.backend) is None:
            return 'master'
        return find_manifest(self.config.backend, None).default_ref

    def _existing_checkout(self, src: Path) -> None:
        """Apply ``--update`` / ``--ref`` to a checkout that already exists."""
        cfg = self.config
        if not cfg.update and (cfg.ref is None or self._at_ref(src, cfg.ref)):
            return
        try:
            dirty = self.git('status', '--porcelain', cwd=src, capture=True)
        except subprocess.CalledProcessError as err:
            msg = f'git status failed in {src}: {(err.stderr or "").strip()}'
            raise BuildError(msg) from None
        if dirty:
            msg = f'{src} has local changes; commit or stash them before using --ref or --update.'
            raise BuildError(msg)
        if cfg.update:
            self.git('fetch', '--tags', 'origin', cwd=src)
        ref = cfg.ref
        if ref is None:
            ref = find_manifest(cfg.backend, src).default_ref
        if cfg.update or self.confirm(
            f'Check out {ref!r} in {src}? (n keeps the commit that is checked out)'
        ):
            self.git('-c', 'advice.detachedHead=false', 'checkout', ref, cwd=src)

    def _at_ref(self, src: Path, ref: str) -> bool:
        """Return whether a checkout's HEAD is already the commit ``ref`` names."""
        try:
            head = self.git('rev-parse', 'HEAD', cwd=src, capture=True)
            target = self.git('rev-parse', f'{ref}^{{commit}}', cwd=src, capture=True)
        except subprocess.CalledProcessError:
            return False
        return head == target

    def phase_tools(self) -> None:
        """Check every tool, then install what is missing after one confirmation."""
        cfg = self.config
        self.step('2/8 Tools')
        tools = _tools.required_tools(cfg.os_name, self.manifest.tools, repair=cfg.repair)
        self._activate_msvc(tools)
        table = Table(show_header=True, header_style='bold purple')
        table.add_column('Tool')
        table.add_column('Found')
        table.add_column('Status')
        missing_pip: list[_tools.Tool] = []
        blocking: list[_tools.Tool] = []
        for tool in tools:
            ok = _tools.detect(tool, self.env)
            found = f'{tool.path} ({tool.version})' if tool.version else (tool.path or '-')
            if ok:
                status = '[green]ok[/green]'
            elif tool.pip:
                status = (
                    '[yellow]missing[/yellow]' if tool.path is None else '[yellow]too old[/yellow]'
                )
                missing_pip.append(tool)
            else:
                status = '[red]missing[/red]'
                blocking.append(tool)
            table.add_row(tool.label, found, status)
        missing_packages = _tools.python_build_packages()
        for name in _tools.PYTHON_BUILD_PACKAGES:
            if name in missing_packages:
                table.add_row(f'{name} (python)', '-', '[yellow]missing[/yellow]')
            else:
                table.add_row(f'{name} (python)', sys.prefix, '[green]ok[/green]')
        table.add_row(*self._compiler_cache_row())
        self.console.print(table)
        if blocking:
            lines = [f'{t.name}: {t.instructions}' for t in blocking]
            msg = 'Install these and run again:\n  ' + '\n  '.join(lines)
            raise BuildError(msg)

        specs = [f'{t.pip}{t.spec}' for t in missing_pip] + missing_packages
        cmd = _tools.pip_install_command(*specs) if specs else None
        if specs and cmd is None:
            msg = (
                f'Install {", ".join(specs)} into {sys.prefix} and run again (no pip or uv found).'
            )
            raise BuildError(msg)
        if cfg.install and _tools.pip_install_command() is None:
            msg = '--install needs pip or uv in this environment.'
            raise BuildError(msg)
        self.confirm_installs(self.install_plan(specs))
        if cmd is not None:
            self.run(cmd)
            if not cfg.dry_run:
                bad = [t.label for t in missing_pip if not _tools.detect(t, self.env)]
                if bad:
                    msg = f'Still not usable after install: {", ".join(bad)}.'
                    raise BuildError(msg)
        self._activate_compiler_cache()

    def _activate_msvc(self, tools: list[_tools.Tool]) -> None:
        """On Windows, import the MSVC environment when ``cl`` is not already on PATH."""
        if self.config.os_name != 'windows':
            return
        msvc = next(t for t in tools if t.name == 'msvc')
        if _tools.detect(msvc, self.env):
            return
        vcvarsall = _tools.find_vcvarsall()
        if vcvarsall is None:
            return
        self.say(
            f'Visual Studio found; importing its build environment from [bold]{vcvarsall}[/bold].'
        )
        if (
            self.confirm('Import it? (n keeps the current environment)')
            and not self.config.dry_run
        ):
            try:
                self.env = _tools.msvc_environment(vcvarsall, env=self.env)
            except subprocess.CalledProcessError as err:
                msg = f'{vcvarsall} failed with exit status {err.returncode}.'
                raise BuildError(msg) from None

    def _compiler_cache_row(self) -> tuple[str, str, str]:
        """Return the tools-table row for the ``--compiler-cache`` choice."""
        choice = self.config.compiler_cache
        if choice == 'none':
            return 'compiler cache', '-', '[dim]off (--compiler-cache none)[/dim]'
        tool = _tools.compiler_cache_tool(choice, self.config.os_name, self.env)
        if tool is None:
            label = 'ccache/sccache' if choice == 'auto' else choice
            status = (
                '[yellow]missing, optional[/yellow]' if choice == 'auto' else '[red]missing[/red]'
            )
            return f'{label} (compiler cache)', '-', status
        found = f'{tool.path} ({tool.version})' if tool.version else (tool.path or '-')
        return f'{tool.name} (compiler cache)', found, '[green]ok[/green]'

    def cache_dir(self, var: str) -> Path:
        """Return the compiler cache directory.

        Parameters
        ----------
        var : str
            Cache tool's directory variable, ``CCACHE_DIR`` or ``SCCACHE_DIR``.

        Returns
        -------
        Path
            ``--cache-dir``, else ``var`` when it is set, else ``<build-dir>/compiler-cache``.

        """
        if self.config.cache_dir is None and self.env.get(var):
            return Path(self.env[var])
        return self.config.compiler_cache_dir

    def _activate_compiler_cache(self) -> None:
        """Pick the compiler launcher and point it at the cache directory."""
        cfg = self.config
        if cfg.compiler_cache == 'none':
            self.say('Compiler cache: off')
            return
        self.launcher = _tools.compiler_cache_tool(cfg.compiler_cache, cfg.os_name, self.env)
        if self.launcher is None:
            hint = _tools.compiler_cache_instructions(cfg.os_name)
            if cfg.compiler_cache != 'auto':
                msg = f'{cfg.compiler_cache} was not found on PATH. Install it: {hint}'
                raise BuildError(msg)
            self.say(
                '[yellow]No ccache/sccache on PATH; rebuilds will be slow. '
                f'Install one: {hint}[/yellow]'
            )
            return
        var = 'SCCACHE_DIR' if self.launcher.name == 'sccache' else 'CCACHE_DIR'
        cache_dir = self.cache_dir(var)
        self.say(f'Compiler cache: [bold]{self.launcher.name}[/bold] in [bold]{cache_dir}[/bold]')
        self.env[var] = str(cache_dir)
        if self.launcher.name == 'ccache':
            if cfg.os_name != 'windows':
                self.env.setdefault('CCACHE_COMPILERCHECK', '%compiler% --version')
            base = Path(os.path.commonpath([cfg.src_dir.resolve(), cfg.cmake_dir.resolve()]))
            if base != Path(base.anchor):
                self.env.setdefault('CCACHE_BASEDIR', str(base))
                self.env.setdefault('CCACHE_NOHASHDIR', '1')

    def install_plan(self, specs: Sequence[str]) -> list[str]:
        """Return one line per thing the build installs.

        Parameters
        ----------
        specs : Sequence[str]
            Missing Python packages with their version requirements.

        Returns
        -------
        list[str]
            What gets installed, and where.

        """
        items = [f'{spec} into {sys.prefix}' for spec in specs]
        if self.config.install:
            items.append(f'the {self.config.backend} wheel into {sys.prefix}, once it is built')
        return items

    def phase_groups(self) -> None:
        """Show the groups and options, letting the user adjust them interactively."""
        cfg = self.config
        self.step('3/8 Build options')
        self._resolve_selection()
        self._print_selection()
        if not cfg.prompting:
            self.say(
                'Change the selection with --with/--without/--all/--minimal and --opt NAME=VALUE.'
            )
            return
        self.say(
            'Enter or y builds with these. n asks about each group and option in turn. '
            'Flags change the selection directly, e.g. [bold]--without web --opt smp=TBB[/bold] '
            '(--with, --without, --all, --minimal, --opt NAME=VALUE). Ctrl-C stops.'
        )
        self._ask_until_yes(
            'Build with these? [y / n / flags]',
            apply=self._apply_selection_flags,
            resolve=self._resolve_selection,
            show=self._print_selection,
            walk=self._adjust_selection,
        )

    def _ask_until_yes(
        self,
        question: str,
        *,
        apply: Callable[[str], None],
        resolve: Callable[[], None],
        show: Callable[[], None],
        walk: Callable[[], None],
    ) -> None:
        """Ask until yes: ``n`` walks through the settings one by one, flags change them."""
        while True:
            answer = Prompt.ask(question, default='y', console=self.console).strip()
            if answer.lower() in ('y', 'yes'):
                return
            saved = (copy.deepcopy(self.config), self.launcher, dict(self.env))
            try:
                if answer.lower() in ('n', 'no'):
                    walk()
                else:
                    apply(answer)
                resolve()
            except BuildError as err:
                config, self.launcher, self.env = saved
                for name in (f.name for f in fields(config)):
                    setattr(self.config, name, getattr(config, name))
                resolve()
                self.say(f'[red]{escape(str(err))}[/red]')
                continue
            show()

    def _resolve_selection(self) -> None:
        """Resolve the groups and options from the configuration."""
        cfg = self.config
        try:
            self.groups = resolve_groups(
                self.manifest,
                cfg.os_name,
                with_=cfg.with_,
                without=cfg.without,
                all_=cfg.all_,
                minimal=cfg.minimal,
            )
            self.option_values = resolve_options(self.manifest, cfg.options)
        except ManifestError as err:
            raise BuildError(str(err)) from None

    @staticmethod
    def _parse_flags(text: str, switches: set[str], valued: set[str]) -> list[tuple[str, str]]:
        """Split a prompt answer into ``(flag, value)`` pairs.

        Parameters
        ----------
        text : str
            Answer typed at the prompt.

        switches : set[str]
            Flags that take no value.

        valued : set[str]
            Flags that take one value, inline after ``=`` or as the next word.

        Returns
        -------
        list[tuple[str, str]]
            Flags in order, switches with an empty value.

        """
        try:
            words = shlex.split(text)
        except ValueError as err:
            msg = f'Cannot parse {text!r}: {err}'
            raise BuildError(msg) from None
        pairs: list[tuple[str, str]] = []
        i = 0
        while i < len(words):
            flag, sep, inline = words[i].partition('=')
            if '-D' in valued and words[i].startswith('-D') and len(words[i]) > 2:
                pairs.append(('-D', words[i][2:]))
            elif flag in switches:
                pairs.append((flag, ''))
            elif flag in valued:
                if not sep:
                    i += 1
                    if i == len(words):
                        msg = f'{flag} needs a value.'
                        raise BuildError(msg)
                    inline = words[i]
                pairs.append((flag, inline))
            else:
                msg = f'Unknown flag {words[i]!r}. Use {", ".join(sorted(switches | valued))}.'
                raise BuildError(msg)
            i += 1
        return pairs

    def _apply_selection_flags(self, text: str) -> None:
        """Apply selection flags typed at the prompt to the configuration."""
        cfg = self.config
        pairs = self._parse_flags(text, {'--all', '--minimal'}, {'--with', '--without', '--opt'})
        for flag, value in pairs:
            names = [n for n in value.split(',') if n]
            if flag == '--all':
                cfg.all_, cfg.minimal = True, False
            elif flag == '--minimal':
                cfg.all_, cfg.minimal = False, True
            elif flag == '--with':
                cfg.with_ += [n for n in names if n not in cfg.with_]
                cfg.without = [n for n in cfg.without if n not in names]
            elif flag == '--without':
                cfg.without += [n for n in names if n not in cfg.without]
                cfg.with_ = [n for n in cfg.with_ if n not in names]
            else:
                key = value.partition('=')[0]
                cfg.options = [o for o in cfg.options if o.partition('=')[0] != key] + [value]

    def _print_selection(self) -> None:
        """Print the group and option table."""
        table = Table(show_header=True, header_style='bold purple')
        table.add_column('Group / option')
        table.add_column('Value')
        table.add_column('Description')
        for group in self.manifest.groups_for(self.config.os_name):
            on = group.name in self.groups
            value = '[green]on[/green]' if on else '[dim]off[/dim]'
            if group.required:
                value += ' (required)'
            table.add_row(group.name, value, group.help)
        for option in self.manifest.options.values():
            value = str(self.option_values[option.name])
            table.add_row(f'--opt {option.name}', value, option.help)
        self.console.print(table)

    def _adjust_selection(self) -> None:
        """Ask about every group and option in turn, writing the answers to the configuration."""
        cfg = self.config
        chosen: list[str] = []
        for group in self.manifest.groups_for(self.config.os_name):
            clash = sorted(set(group.conflicts) & set(chosen))
            if clash:
                self.say(f'[dim]{group.name}: off, conflicts with {clash}[/dim]')
                continue
            if group.required or Confirm.ask(
                f'Enable {group.name!r} ({group.help})?',
                default=group.name in self.groups,
                console=self.console,
            ):
                chosen.append(group.name)
        options = []
        for option in self.manifest.options.values():
            raw = Prompt.ask(
                f'{option.name} ({option.help})',
                default=str(self.option_values[option.name]),
                choices=list(option.choices) or None,
                console=self.console,
            )
            options.append(f'{option.name}={raw}')
        cfg.with_, cfg.without, cfg.all_, cfg.minimal = chosen, [], False, True
        cfg.options = options

    def version_suffix(self) -> str | None:
        """Return the wheel version suffix.

        Returns
        -------
        str | None
            ``--version-suffix`` when given, nothing on a tag, else ``dev0+g<sha>``.

        """
        cfg = self.config
        if cfg.version_suffix is not None:
            return cfg.version_suffix or None
        if self.manifest.version_command:
            if cfg.dry_run:
                return '<version_command>'
            command = self._expand(self.manifest.version_command)
            try:
                out = subprocess.run(
                    command, cwd=cfg.src_dir, capture_output=True, text=True, check=True
                ).stdout
            except subprocess.CalledProcessError as err:
                msg = f'version_command failed with exit status {err.returncode}: {err.stderr}'
                raise BuildError(msg) from None
            except OSError as err:
                msg = f'Cannot run version_command {command[0]}: {err}'
                raise BuildError(msg) from None
            return out.strip() or None
        if not (cfg.src_dir / '.git').exists():
            ref = cfg.ref or self.manifest.default_ref
            if cfg.dry_run and not (ref and _TAG_LIKE.match(ref)):
                return 'dev0+g<sha>'
            return None
        return self._git_version_suffix()

    def _git_version_suffix(self) -> str | None:
        """Return ``dev0+g<sha>`` for the checkout, or ``None`` on a tag or without git."""
        cfg = self.config
        try:
            self.git('describe', '--exact-match', '--tags', 'HEAD', cwd=cfg.src_dir, capture=True)
        except subprocess.CalledProcessError:
            pass
        else:
            return None
        try:
            sha = self.git('rev-parse', '--short=8', 'HEAD', cwd=cfg.src_dir, capture=True)
        except subprocess.CalledProcessError:
            return None
        return f'dev0+g{sha}'

    def configure_command(self) -> list[str]:
        """Assemble the ``cmake`` configure command.

        Returns
        -------
        list[str]
            Command and arguments.

        """
        cfg = self.config
        cmd = ['cmake', '-Wno-dev', '-S', str(cfg.src_dir), '-B', str(cfg.cmake_dir), '-GNinja']
        cmd += [f'-U{key}' for key in cmake_unset(self.manifest, cfg.os_name, groups=self.groups)]
        init_cache = self.manifest.init_cache.get(cfg.os_name)
        if init_cache:
            cmd += ['-C', str(cfg.src_dir / init_cache)]
        cmd += [
            f'-DCMAKE_BUILD_TYPE={cfg.build_type}',
            f'-DPython3_EXECUTABLE={self.python}',
            '-DPython3_FIND_STRATEGY=LOCATION',
        ]
        launcher = self.launcher.name if self.launcher is not None else ''
        cmd += [
            f'-DCMAKE_C_COMPILER_LAUNCHER={launcher}',
            f'-DCMAKE_CXX_COMPILER_LAUNCHER={launcher}',
        ]
        cmd.append(f'-DVTK_VERSION_SUFFIX={self.version_suffix() or ""}')
        entries = cmake_entries(
            self.manifest, cfg.os_name, groups=self.groups, options=self.option_values
        )
        cmd += [f'-D{e}' for e in entries]
        cmd += [f'-D{e}' for e in cfg.defines]
        return cmd

    def configure_env(self) -> dict[str, str]:
        """Return the environment cmake runs in: the manifest options and the wheel platform.

        Returns
        -------
        dict[str, str]
            Variable name to value.

        """
        return {**option_environment(self.manifest, self.option_values), **self.wheel_env()}

    def _stored_state(self) -> object:
        """Return the configure state saved by the last run, or ``None`` when unreadable."""
        state_file = self.config.cmake_dir / STATE_FILENAME
        try:
            return json.loads(state_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None

    def phase_configure(self) -> None:
        """Run cmake, or skip it when the tree is already configured the same way."""
        cfg = self.config
        self.step('4/8 Configure')
        if cfg.clean and cfg.cmake_dir.exists():
            self.say(f'--clean removes [bold]{cfg.cmake_dir}[/bold].')
            if self.confirm('Delete it? (n keeps it and configures in place)') and not cfg.dry_run:
                shutil.rmtree(cfg.cmake_dir)
        cmd = self.configure_command()
        env = self.configure_env()
        state = {'command': cmd, 'env': env}
        configured = (cfg.cmake_dir / 'CMakeCache.txt').exists()
        stored = self._stored_state() if configured else None
        if configured and not cfg.reconfigure and stored == state:
            self.say(
                f'[bold]{cfg.cmake_dir}[/bold] is already configured; skipping '
                '(use --reconfigure).'
            )
            return
        if configured and stored != state:
            if self._only_version_changed(stored, cmd, env):
                self.say('The source is at a new commit, so the version suffix changes.')
            else:
                self.say('The configure command differs from the one this tree was set up with.')
                if not self.confirm('Reconfigure? (n stops)'):
                    msg = 'Stopped; the build tree keeps its earlier configuration.'
                    raise BuildError(msg)
        self._print_configure_settings()
        if cfg.prompting:
            self.say(
                'Enter or y configures with these. n asks about each setting in turn. '
                'Flags change a setting directly, e.g. '
                '[bold]--build-type Debug -D VTK_ENABLE_LOGGING=OFF[/bold] '
                '(--build-type, --compiler-cache, --version-suffix, -D KEY=VALUE). Ctrl-C stops.'
            )
            self._ask_until_yes(
                'Run cmake with these? [y / n / flags]',
                apply=self._apply_configure_flags,
                resolve=lambda: None,
                show=self._print_configure_settings,
                walk=self._walk_configure_settings,
            )
            cmd = self.configure_command()
            env = self.configure_env()
            state = {'command': cmd, 'env': env}
        elif not self.confirm('Run cmake with these? (n stops)'):
            msg = 'Stopped before configure.'
            raise BuildError(msg)
        self.run(cmd, env=env)
        if not cfg.dry_run:
            state_file = cfg.cmake_dir / STATE_FILENAME
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text(json.dumps(state, indent=2), encoding='utf-8')
        self.hooks('post_configure')

    def _print_configure_settings(self) -> None:
        """Print the settings the cmake command is assembled from."""
        cfg = self.config
        table = Table(show_header=True, header_style='bold purple')
        table.add_column('Setting')
        table.add_column('Value')
        table.add_column('Flag')
        launcher = self.launcher.name if self.launcher is not None else 'none'
        table.add_row('Build type', cfg.build_type, '--build-type')
        table.add_row('Build tree', str(cfg.cmake_dir), '--build-dir')
        table.add_row('Python', str(self.python), '')
        table.add_row('Compiler cache', launcher, '--compiler-cache')
        table.add_row('Version suffix', self.version_suffix() or '(release)', '--version-suffix')
        defines = escape(shlex.join(cfg.defines)) or '(none)'
        table.add_row('Extra cache entries', defines, '-D KEY=VALUE')
        self.console.print(table)

    def _walk_configure_settings(self) -> None:
        """Ask for each configure setting in turn."""
        cfg = self.config
        cfg.build_type = Prompt.ask(
            'Build type', default=cfg.build_type, choices=list(BUILD_TYPES), console=self.console
        )
        cache = Prompt.ask(
            'Compiler cache',
            default=cfg.compiler_cache,
            choices=list(COMPILER_CACHES),
            console=self.console,
        )
        if cache != cfg.compiler_cache:
            self._apply_configure_flags(f'--compiler-cache {cache}')
        cfg.version_suffix = Prompt.ask(
            'Version suffix, e.g. dev0+gabc12345 (empty for a release version)',
            default=self.version_suffix() or '',
            console=self.console,
        )
        raw = Prompt.ask(
            'Extra CMake cache entries, KEY=VALUE separated by spaces (empty for none)',
            default=shlex.join(cfg.defines),
            console=self.console,
        )
        try:
            cfg.defines = shlex.split(raw)
        except ValueError as err:
            msg = f'Cannot parse {raw!r}: {err}'
            raise BuildError(msg) from None

    def _apply_configure_flags(self, text: str) -> None:
        """Apply configure flags typed at the prompt to the configuration."""
        cfg = self.config
        valued = {'--build-type', '--version-suffix', '--compiler-cache', '-D', '--define'}
        for flag, value in self._parse_flags(text, set(), valued):
            if flag == '--build-type':
                if value not in BUILD_TYPES:
                    msg = f'--build-type must be one of {", ".join(BUILD_TYPES)}.'
                    raise BuildError(msg)
                cfg.build_type = value
            elif flag == '--version-suffix':
                cfg.version_suffix = value
            elif flag == '--compiler-cache':
                if value not in COMPILER_CACHES:
                    msg = f'--compiler-cache must be one of {", ".join(COMPILER_CACHES)}.'
                    raise BuildError(msg)
                cfg.compiler_cache = value
                self.launcher = None
                self._activate_compiler_cache()
            else:
                key = value.partition('=')[0]
                cfg.defines = [d for d in cfg.defines if d.partition('=')[0] != key] + [value]

    @staticmethod
    def _only_version_changed(stored: object, cmd: list[str], env: dict[str, str]) -> bool:
        """Return whether a stored state differs from this one only in the version suffix."""
        if not isinstance(stored, dict) or stored.get('env') != env:
            return False
        old = stored.get('command')
        if not isinstance(old, list) or len(old) != len(cmd):
            return False
        prefix = '-DVTK_VERSION_SUFFIX='
        return all(
            a == b or (str(a).startswith(prefix) and b.startswith(prefix))
            for a, b in zip(old, cmd, strict=True)
        )

    def phase_build(self) -> None:
        """Compile with ninja through cmake."""
        cfg = self.config
        self.step('5/8 Build')
        cores = os.cpu_count() or 1
        jobs = cfg.jobs or cores
        flag = ' (--jobs)' if cfg.jobs else ''
        self.say(
            f'{cores} CPU cores detected; compiling with [bold]{jobs}[/bold] jobs{flag}. '
            'A full VTK build takes tens of minutes to hours.'
        )
        jobs = self._ask_jobs(jobs)
        self.run(
            ['cmake', '--build', str(cfg.cmake_dir), '--parallel', str(jobs)],
            log=cfg.build_dir / 'build.log',
        )
        if self.launcher is not None:
            self.run([self.launcher.name, '--show-stats'], fatal=False)
        self.hooks('post_build')

    def _ask_jobs(self, jobs: int) -> int:
        """Confirm the build, taking a number as the job count instead of a yes.

        Parameters
        ----------
        jobs : int
            Proposed job count.

        Returns
        -------
        int
            Job count to build with.

        """
        if not self.config.prompting:
            if not self.confirm('Start the build? (n stops)'):
                msg = 'Stopped before building.'
                raise BuildError(msg)
            return jobs
        while True:
            answer = (
                Prompt.ask(
                    'Start the build? [y / n = stop / a number to use that many jobs]',
                    default='y',
                    console=self.console,
                )
                .strip()
                .lower()
            )
            if answer in ('y', 'yes'):
                return jobs
            if answer in ('n', 'no'):
                msg = 'Stopped before building.'
                raise BuildError(msg)
            if answer.isdigit() and int(answer) > 0:
                return int(answer)
            self.say('[red]Answer y, n or a positive number.[/red]')

    def wheel_env(self) -> dict[str, str]:
        """Return the extra environment for ``bdist_wheel``.

        Returns
        -------
        dict[str, str]
            Platform tag variables on macOS; empty elsewhere.

        """
        cfg = self.config
        if cfg.os_name != 'macos':
            return {}
        arch = platform.machine()
        prefix = 'CMAKE_OSX_DEPLOYMENT_TARGET='
        targets = [
            e[len(prefix) :] for e in self.manifest.cmake_os['macos'] if e.startswith(prefix)
        ]
        target = targets[0] if targets else '11.0'
        return {'_PYTHON_HOST_PLATFORM': f'macosx-{target}-{arch}', 'ARCHFLAGS': f'-arch {arch}'}

    def phase_wheel(self) -> None:
        """Package the build tree into a wheel."""
        cfg = self.config
        self.step('6/8 Wheel')
        self.say(f'The wheel is written to [bold]{cfg.dist_dir}[/bold].')
        if not self.confirm('Build the wheel? (n stops)'):
            msg = 'Stopped before packaging.'
            raise BuildError(msg)
        self.run([self.python, 'setup.py', 'bdist_wheel'], cwd=cfg.cmake_dir, env=self.wheel_env())
        self.hooks('post_wheel')
        self.wheel = self._newest_wheel(cfg.dist_dir)

    def _newest_wheel(self, folder: Path) -> Path | None:
        """Return the most recently written wheel in a folder."""
        wheels = sorted(folder.glob('*.whl'), key=lambda p: p.stat().st_mtime)
        if not wheels:
            if self.config.dry_run:
                return None
            msg = f'No wheel found in {folder}.'
            raise BuildError(msg)
        return wheels[-1]

    def repair_command(self, wheel: Path) -> tuple[list[str], dict[str, str]]:
        """Assemble the repair command for this operating system.

        Parameters
        ----------
        wheel : Path
            Wheel to repair.

        Returns
        -------
        list[str]
            Command and arguments.

        dict[str, str]
            Extra environment for the command.

        """
        cfg = self.config
        out = str(cfg.wheelhouse)
        if cfg.os_name == 'linux':
            return ['auditwheel', 'repair', str(wheel), '-w', out], {}
        if cfg.os_name == 'windows':
            bin_dir = str(cfg.cmake_dir / 'bin')
            return ['delvewheel', 'repair', '--add-path', bin_dir, '-w', out, str(wheel)], {}
        dylibs = cfg.cmake_dir.rglob('libvtksys*.dylib')
        libdir = next((p.parent for p in dylibs), cfg.cmake_dir / 'lib')
        arch = platform.machine()
        return (
            ['delocate-wheel', '--require-archs', arch, '-w', out, '-v', str(wheel)],
            {'DYLD_LIBRARY_PATH': str(libdir)},
        )

    def phase_repair(self) -> None:
        """Bundle the shared libraries into the wheel."""
        cfg = self.config
        self.step('7/8 Repair')
        if not cfg.repair:
            self.say('Skipped (--no-repair).')
            return
        self.say(
            'Repairing bundles the libraries the wheel links against from outside it, '
            'so it installs on other machines.'
        )
        if cfg.os_name == 'linux':
            self.say(
                "[dim]On Linux the wheel is tagged for this machine's glibc; portable manylinux "
                'wheels need a manylinux container.[/dim]'
            )
        if not self.confirm('Repair the wheel? (n skips it)'):
            self.say('Skipped.')
            return
        wheel = self.wheel or cfg.dist_dir / f'{cfg.backend}-*.whl'
        cmd, env = self.repair_command(wheel)
        self.run(cmd, env=env)
        if not cfg.dry_run:
            self.wheel = self._newest_wheel(cfg.wheelhouse)

    def install_command(self, wheel: Path) -> list[str] | None:
        """Assemble the command that installs a wheel into this interpreter.

        Parameters
        ----------
        wheel : Path
            Wheel to install.

        Returns
        -------
        list[str] | None
            Command, or ``None`` when neither pip nor uv is available.

        """
        return _tools.pip_install_command(str(wheel), args=('--no-deps', '--force-reinstall'))

    def phase_install(self) -> None:
        """Install the wheel into the current environment when asked."""
        cfg = self.config
        self.step('8/8 Install')
        wheel = self.wheel or (cfg.wheelhouse if cfg.repair else cfg.dist_dir) / (
            f'{cfg.backend}-*.whl'
        )
        cmd = self.install_command(wheel)
        if cmd is None:
            self.say(f'No pip or uv found; install the wheel yourself: {wheel}')
            return
        wanted = cfg.install or self.confirm(
            f'Install {wheel.name} into {sys.prefix}? (n leaves it in the wheelhouse)',
            default=False,
        )
        if not wanted:
            self.say(f'Not installed. To install later:\n  [cyan]{escape(_join(cmd))}[/cyan]')
            return
        self.run(cmd)

    def build_in_manylinux(self) -> Path | None:
        """Run the build inside a manylinux container, then install on the host.

        Returns
        -------
        Path or None
            The repaired wheel, or ``None`` on a dry run.

        """
        cfg = self.config
        image = cfg.manylinux or _manylinux.default_image()
        if cfg.compiler_cache == 'sccache':
            msg = '--manylinux builds use ccache; pass --compiler-cache ccache, auto or none.'
            raise BuildError(msg)
        self.say(
            f'The build runs inside the [bold]{image}[/bold] container through Docker, so '
            'the wheel is portable across Linux machines. The source is prepared here, then '
            'the container installs the build dependencies and runs the remaining steps with '
            f'[bold]{cfg.build_dir}[/bold] mounted as its build tree.'
        )
        docker = _tools.Tool('docker', ('docker',), instructions=_manylinux.DOCKER_INSTRUCTIONS)
        if not _tools.detect(docker, self.env):
            msg = f'docker was not found on PATH. {docker.instructions}'
            raise BuildError(msg)
        if not cfg.dry_run:
            probe = subprocess.run([str(docker.path), 'info'], capture_output=True, check=False)
            if probe.returncode != 0:
                msg = 'Docker is installed but its daemon is not running or not reachable.'
                raise BuildError(msg)
        self.confirm_start()
        self.phase_source()
        install_here = cfg.install and cfg.os_name == 'linux'
        if install_here and _tools.pip_install_command() is None:
            msg = '--install needs pip or uv in this environment.'
            raise BuildError(msg)
        if install_here:
            self.confirm_installs(self.install_plan([]))
        mounts: dict[str, Path] = {_manylinux.WORK: cfg.build_dir, _manylinux.SOURCE: cfg.src_dir}
        if cfg.compiler_cache != 'none':
            mounts[_manylinux.CACHE] = self.cache_dir('CCACHE_DIR')
        package_mount = f'{_manylinux.PYVISTA_ROOT}/pyvista'
        script = _manylinux.container_script(
            python=_manylinux.container_python(),
            packages=self.manifest.manylinux_packages,
            requirements=_manylinux.pyvista_requirements(),
            config=json.dumps(self.container_config()),
            ccache=cfg.compiler_cache != 'none',
            owner=_manylinux.file_owner(),
            paths=list(mounts),
        )
        mounts[package_mount] = _manylinux.pyvista_package()
        cmd = _manylinux.docker_command(
            image, mounts=mounts, script=script, interactive=cfg.prompting, env=self.env
        )
        if not self.confirm('Run the build in the container? (n stops)'):
            msg = 'Stopped before the container build.'
            raise BuildError(msg)
        if not cfg.dry_run:
            for inside, folder in mounts.items():
                if inside != package_mount:
                    folder.mkdir(parents=True, exist_ok=True)
        self.run(cmd)
        if not cfg.dry_run:
            self.wheel = self._newest_wheel(cfg.wheelhouse if cfg.repair else cfg.dist_dir)
        if cfg.os_name == 'linux':
            self.phase_install()
        else:
            self.say(
                f'The wheel targets Linux, so it is not installed on this {cfg.os_name} host.'
            )
        self.summary()
        return self.wheel

    def container_config(self) -> dict[str, object]:
        """Return the container's build configuration, with host paths mapped to the mounts.

        Returns
        -------
        dict[str, object]
            :class:`BuildConfig` fields for ``python -m pyvista._build``.

        """
        cfg = self.config
        cache = cfg.compiler_cache != 'none'
        return {
            'backend': cfg.backend,
            'build_dir': _manylinux.WORK,
            'source': _manylinux.SOURCE,
            'with_': cfg.with_,
            'without': cfg.without,
            'all_': cfg.all_,
            'minimal': cfg.minimal,
            'build_type': cfg.build_type,
            'jobs': cfg.jobs,
            'defines': cfg.defines,
            'options': cfg.options,
            'clean': cfg.clean,
            'reconfigure': cfg.reconfigure,
            'version_suffix': cfg.version_suffix,
            'repair': cfg.repair,
            'install': False,
            'yes': True,
            'interactive': cfg.prompting,
            'os_name': 'linux',
            'compiler_cache': cfg.compiler_cache,
            'cache_dir': _manylinux.CACHE if cache else None,
        }

    def summary(self) -> None:
        """Print where things ended up."""
        cfg = self.config
        self.console.rule('[bold green]Done')
        if self.wheel is not None:
            self.say(f'Wheel: [bold]{self.wheel}[/bold]')
        self.say(f'Source: {cfg.src_dir}\nBuild tree: {cfg.cmake_dir}')
        self.say('Edit the source and run the same command again for an incremental rebuild.')
