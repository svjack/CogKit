SHELL = /bin/sh

.PHONY: check

check:
	hatch fmt --check
	hatch run types:check
