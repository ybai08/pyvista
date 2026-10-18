.. _building_vtk:

Building VTK
============
Kitware provides Python wheels for VTK at `PyPI VTK
<https://pypi.org/project/vtk/>`_, but there are situations where you
may need to build VTK from source: a new release of Python, EGL or
OSMesa rendering, modules the stock wheel leaves out, a VTK branch with
a fix you need, or a backend such as `cvista
<https://github.com/pyvista/cvista>`_.

PyVista ships a command that does this for you:

.. code-block:: bash

    pyvista build vtk

.. versionadded:: 0.50

It works on Linux, macOS, and Windows and follows the official `Building
Python wheels
<https://docs.vtk.org/en/latest/advanced/build_python_wheels.html>`_
instructions, with the options PyVista's own CI uses. The wheel is built
for the Python that runs the command. By default it
says where the build goes and asks once before starting, then builds
with those defaults straight through, printing every command as it
runs. Anything it has to install is listed and confirmed too, and
``--yes`` answers both questions in advance. Add ``--interactive`` to
walk through the build one step at a
time: each step shows its settings, says what it is about to do and
where, and waits for a yes. Press Enter to accept the default, answer
``n`` to go through the settings one by one, or type the same flags the
command line takes to change one in place.

.. code-block:: bash

    pyvista build vtk --interactive

The command reference with more examples is at :ref:`cli_build`. The
result is a wheel, so the build selects itself through the usual
mechanism once installed: see :envvar:`PYVISTA_VTK_BACKEND` and
:ref:`the backend section of the installation guide <vtk_backend>`.

The Steps
~~~~~~~~~
1. **Tools.** ``git``, ``cmake``, ``ninja``, a C++ compiler, and the
   wheel repair tool are checked before anything is downloaded (see
   :ref:`build_tooling` for what each one does). Everything the build
   would install is then listed, each item with where it goes:
   ``cmake``, ``ninja``, and the
   repair tool when missing or too old, ``setuptools`` and ``wheel``, and
   the finished wheel with ``--install``. One confirmation covers the
   whole list, and ``--yes`` answers it in advance. Without a terminal
   and without ``--yes`` the command stops instead of installing. A
   missing compiler stops with instructions for your operating system.
   On Windows, Visual Studio's build environment is imported
   automatically. ``--compiler-cache auto`` adds a compiler cache
   (``ccache`` or ``sccache``) when one is on ``PATH``. When the cloned
   source asks for other ``cmake`` or ``ninja`` versions than the ones
   checked, as on a first ``cvista`` build, those are checked again
   after step 2.
2. **Source.** VTK is cloned into ``./vtk-build/src`` at the release tag
   PyVista supports. Use ``--ref`` for another tag, branch, or commit,
   ``--build-dir`` to put everything somewhere else, and ``--source DIR``
   to build a checkout you already have. An existing checkout is reused
   as is, local edits included.
3. **Build options.** On top of the modules every build has, the
   modules to build are organized into named groups. The default set
   matches the stock wheel plus what PyVista's CI enables:

   - ``rendering``: OpenGL rendering, widgets, charts, and text
   - ``io``: extra readers and writers (scene import, Exodus, Xdmf2)
   - ``web``: web export modules used by trame and the Jupyter backends
   - ``matplotlib``: Matplotlib-rendered math text and labels
   - ``parallel``: DIY2-based parallel filters
   - ``serialization``: object serialization and ``.pyi`` stub files
   - ``headless-egl`` and ``headless-osmesa`` (Linux, off by default):
     off-screen rendering, see :ref:`gpu_off_screen`

   ``--opt smp=VALUE`` picks the threading backend for SMP-enabled
   filters: ``Sequential`` (the default, as in the PyPI wheels),
   ``STDThread``, ``OpenMP``, or ``TBB``, which needs TBB installed.
   ``--all`` enables every group available on your system, leaving out
   off-by-default groups that conflict with an enabled one. At the
   prompt, ``n`` asks about each group in turn, and the command-line
   flags work there as well:

   .. code-block:: bash

       pyvista build vtk --without web,matplotlib
       pyvista build vtk --minimal --with rendering
       pyvista build vtk --all
       pyvista build vtk --opt smp=STDThread

4. **Configure** shows the build type, compiler cache, version suffix,
   and extra CMake cache entries. ``n`` asks about each in turn, and
   ``--build-type``, ``--compiler-cache``, ``--version-suffix``, and
   ``-D`` change one directly. Then ``cmake`` runs. Every command is
   printed in full before it runs.
5. **Build** runs ``ninja`` and logs the full output to ``build.log``.
   It runs one compile job per CPU core, at most one per 2 GB of
   memory, since each job can take that much. ``--jobs`` sets the
   number instead.
6. **Wheel** runs ``setup.py bdist_wheel`` in the build tree.
7. **Repair** bundles the libraries the wheel links against and sets its
   platform tag, so it installs outside the build tree and on other
   machines. ``--no-repair`` skips it.
8. **Install** installs the wheel into the current environment with
   ``--install``, which is part of the confirmation in step 1. Without it,
   ``--interactive`` offers to, and otherwise the wheel's path and the
   ``pip`` command are printed.

Scripted Use
~~~~~~~~~~~~
Without ``--interactive`` the command asks before it starts, and again
before installing anything that is missing. Without a terminal it stops
at the first question, so pass ``--yes`` in CI and scripts to answer
both. ``--dry-run`` prints the commands, and the install list,
without running anything. Optional steps keep their own switches, so
``--no-repair`` skips the repair step and ``--install`` installs the
wheel at the end.

.. code-block:: bash

    pyvista build vtk --ref master --build-type RelWithDebInfo --jobs 8 --yes
    pyvista build vtk --dry-run

Developer Loop
~~~~~~~~~~~~~~
The build tree is kept between runs. Edit files under ``vtk-build/src``
and run ``pyvista build vtk`` again: configure is skipped unless the
options changed, ``ninja`` rebuilds what the edit touched and a new
wheel is written. ``--reconfigure`` forces ``cmake`` to run again,
``--clean`` deletes the build tree first, and ``-D KEY=VALUE`` passes
any CMake cache entry straight through. ``--ref`` leaves a checkout
that is already at that commit as it is, edits included, and otherwise
checks the commit out. ``--update`` fetches first, then checks out
``--ref``, or the release tag when ``--ref`` is not given. Both need a
checkout without local edits when they have to move it:

.. code-block:: bash

    pyvista build vtk --build-type Debug -D VTK_ENABLE_LOGGING=OFF

Other Backends
~~~~~~~~~~~~~~
The same command builds any VTK backend whose source tree carries a
``pyvista-build.toml`` manifest. That file declares the groups and
options the backend offers, its tool requirements, per-platform CMake
init caches, and the extra commands it needs; stock VTK's manifest ships
with PyVista. `cvista <https://github.com/pyvista/cvista>`_, the
community fork described in :ref:`vtk_backend`, is known by
name. A fork of VTK is built as ``vtk`` from its URL, using the manifest
in its tree when it has one and PyVista's otherwise. ``--url`` only
applies when cloning, so give a fork its own ``--build-dir``:

.. code-block:: bash

    pyvista build cvista
    pyvista build vtk --url https://github.com/me/vtk.git --ref my-branch --build-dir vtk-fork-build

A backend built this way is selected like any other installed one:
automatically for ``cvista``, or explicitly through
:envvar:`PYVISTA_VTK_BACKEND`.

.. _gpu_off_screen:

Off-Screen Plotting GPU Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
VTK supports rendering with EGL, enabling rapid off-screen rendering
using GPU hardware acceleration without installing a virtual
framebuffer. The default VTK wheels are not built with this feature. On
Linux, build a wheel with it enabled with:

.. code-block:: bash

    sudo apt install libegl1-mesa-dev  # or: dnf install mesa-libEGL-devel
    pyvista build vtk --with headless-egl

This disables any plotting using the X server, so be prepared to use
this module only on a headless display where you either intend to save
static images or stream the render window to another computer with a
display (for example, using ``pyvista.set_jupyter_backend('server')`` and
JupyterLab). In other words, this wheel will make VTK unusable outside
of an off-screen environment. ``--with headless-osmesa`` builds the
software-rendering equivalent instead.

Portable Linux Wheels
~~~~~~~~~~~~~~~~~~~~~
A wheel built on Linux is tagged for the ``glibc`` of the machine that
built it and may not install elsewhere. To share one, build it inside a
`manylinux <https://github.com/pypa/manylinux>`_ container, as PyVista's
own CI does. ``--manylinux`` does that through `Docker
<https://docs.docker.com/get-docker/>`_, which must be installed and
running:

.. code-block:: bash

    pyvista build vtk --manylinux

The container image matches your machine's architecture
(``manylinux_2_28_x86_64`` or ``manylinux_2_28_aarch64``) and carries
the same Python version as the one running the command. The source is
cloned or updated on your machine first. The container then installs
VTK's build dependencies and PyVista's requirements, mounts the PyVista
you are running read-only, and runs the remaining steps with the source
and ``./vtk-build`` mounted. The repair step runs ``auditwheel`` there,
which gives the wheel its ``manylinux`` tag and puts it in
``vtk-build/wheelhouse``.
Files written by the container are handed back to your user, even when
the build fails. Pass another image with ``--manylinux-image``. With
``--compiler-cache auto`` or ``ccache`` the container installs
``ccache`` and keeps its cache between runs. ``sccache`` is not
available there. A build tree configured inside
the container cannot be continued outside it.
On a Linux host the install step runs afterwards. On macOS and Windows
the wheel is only written, since it is for Linux.

.. _build_tooling:

Tooling
~~~~~~~
What the command looks for and offers to install:

- `git <https://git-scm.com/>`_ clones the source and checks out the
  requested tag, branch, or commit. Never installed by the command.
- `CMake <https://cmake.org/>`_ configures the build: it reads VTK's
  ``CMakeLists.txt`` and the options chosen, finds the compiler and
  Python, and writes the build files. Installed with ``pip`` when
  missing.
- `Ninja <https://ninja-build.org/>`_ is the build tool that CMake
  generates files for; it runs the compiler across all cores and only
  rebuilds what changed. Installed with ``pip`` when missing.
- **A C++ compiler**: GCC or Clang on Linux, Xcode's command-line tools
  on macOS, Visual Studio with the C++ workload on Windows. Never
  installed by the command. A missing one stops with the instructions
  for your operating system.
- **Development headers on Linux** for Python, OpenGL, and X11:
  ``sudo apt install python3-dev libgl1-mesa-dev libx11-dev libxcursor-dev``
  on Debian and Ubuntu, or ``sudo dnf install python3-devel
  mesa-libGL-devel libX11-devel libXcursor-devel`` on Fedora. The tool
  check cannot see them, so a missing one stops ``cmake`` instead.
- `ccache <https://ccache.dev/>`_ and `sccache
  <https://github.com/mozilla/sccache>`_ are compiler caches: they store
  each object file keyed by its inputs, so rebuilding after an edit or a
  reconfigure only recompiles what changed. ``ccache`` is the usual
  choice on Linux and macOS; ``sccache`` is the one that supports MSVC
  on Windows. ``--compiler-cache auto`` uses whichever is on ``PATH``.
  Off by default: within one build tree Ninja already rebuilds only what
  changed, so a cache pays off when the tree is built from scratch, as in
  CI.
- **Wheel repair tools** copy the shared libraries a wheel depends on
  into the wheel itself and set the platform tag, so it installs on
  other machines: `auditwheel
  <https://github.com/pypa/auditwheel>`_ on Linux, `delocate
  <https://github.com/matthew-brett/delocate>`_ on macOS, and `delvewheel
  <https://github.com/adang1345/delvewheel>`_ on Windows. Installed with
  ``pip`` when missing.
- `setuptools <https://setuptools.pypa.io/>`_ and `wheel
  <https://wheel.readthedocs.io/>`_ are the Python packages that
  ``setup.py bdist_wheel`` needs to package the build tree. Installed
  with ``pip`` when missing.
- `Docker <https://docs.docker.com/get-docker/>`_ runs the manylinux
  container for ``--manylinux``. Never installed by the command.

Building by Hand
~~~~~~~~~~~~~~~~
``pyvista build vtk --dry-run`` prints the exact ``cmake`` and
``setup.py`` commands, which is the quickest way to get a starting point
for a manual build. The essential configuration is:

.. code-block:: bash

    git clone https://gitlab.kitware.com/vtk/vtk.git
    cmake -S vtk -B vtk/build -GNinja \
          -DCMAKE_BUILD_TYPE=Release \
          -DVTK_WHEEL_BUILD=ON \
          -DVTK_WRAP_PYTHON=ON \
          -DVTK_BUILD_TESTING=OFF \
          -DVTK_BUILD_DOCUMENTATION=OFF \
          -DVTK_BUILD_EXAMPLES=OFF \
          -DPython3_EXECUTABLE=$(which python)
    cmake --build vtk/build --parallel
    cd vtk/build && python setup.py bdist_wheel

The wheel is written to ``vtk/build/dist``.
