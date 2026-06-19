# conftest.py
import pytest


def pytest_configure(config):
    # Create a place on the pytest config object to store our custom data
    config._comparison_data = {}


@pytest.fixture
def track_diff(request):
    """Fixture to easily log time differences for the current test."""
    test_name = request.node.name

    # We yield a dictionary so the test can inject its timing data
    timing_payload = {}
    yield timing_payload

    # Save the payload to the global config storage after the test finishes
    request.config._comparison_data[test_name] = timing_payload


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Runs at the very end. Standard pytest output printed above this."""
    comparison_data = getattr(config, "_comparison_data", {})

    if not comparison_data:
        return

    # Print a clean, isolated section at the bottom of the terminal
    tr = terminalreporter
    tr.ensure_newline()
    tr.section("PERFORMANCE COMPARISON SUMMARY", sep="=", yellow=True)

    for test_name, data in comparison_data.items():
        tr.write_line(f"📊 {test_name}:")
        for dname, d in data.items():
            lines = ""
            tref = None
            lines += f"For {dname}"
            for n, t in d.items():
                lines += f", {n} took {t:.5f}s"
                if tref is not None:
                    ratio = t / tref
                    lines += f" ({ratio * 100:.4f}% of baseline)"
                else:
                    tref = t
                    lines += " (baseline)"
            tr.write_line(f"\t{lines}.")
            tr.ensure_newline()
