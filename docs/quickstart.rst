Quick Start Guide
=================

These instructions will help you get up and running with *PyChOpMarg* quickly.
It is generally advisable to work within a virtual environment when installing a non-standard package, such as *PyChOpMarg*.
And these instructions will guide you down that path.
This will protect your existing system Python installation from any incompatibilities between it and *PyChOpMarg* and/or its dependencies.

Installation
------------

To install *PyChOpMarg* into a dedicated new virtual environment, execute these commands from an appropriate shell prompt:

1. `python -m venv ~/.venv/pychopmarg`

    - This command creates the new dedicated Python virtual environment.

    * I've assumed that you keep your Python virtual environments in the `.venv/` sub-directory of your home directory. If that's not the case then you'll need to modify the command slightly, to accommodate your system layout and usage habbits.

2. `. ~/.venv/pychopmarg/bin/activate`

    - This command activates the new virtual environment, so we can install packages into it.

    - Windows users should use the command: `. ~/.venv/pychopmarg/Scripts/activate`

    - You'll know that you have correctly activated your new virtual environment when you see the text "(pychopmarg)" appear above your prompt.

3. `pip install pychopmarg`

    - This command installs *PyChOpMarg* and its dependencies into your new virtual environment.

Launching the GUI
-----------------

To launch the *PyChOpMarg* GUI application, simply type: `pychopmarg` and hit `<RETURN>`.
