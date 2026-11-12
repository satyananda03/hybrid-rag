"""Parsing — a file becomes a content list of typed items.

* ``parse``     — ``parse_file``, the one door: picks the backend (``PARSER_BACKEND`` / ``parser_backend``)
* ``router``    — the ``builtin`` backend: the parsers of this package, in this process
* ``external``  — the ``external`` backend: the parser server does the parsing
* ``pdf/contract`` — the content list both backends return
"""
