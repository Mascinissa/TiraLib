from pathlib import Path

import pytest

from tiralib.config import BaseConfig
from tiralib.tiramisu import tiramisu_actions
from tiralib.tiramisu.function_server import ServerExecutionFailedError
from tiralib.tiramisu.schedule import Schedule
from tiralib.tiramisu.tiramisu_program import TiramisuProgram


def _server_program():
    BaseConfig.init()
    cpp_code = Path("examples/function_gemver_MINI_generator.cpp").read_text()
    return TiramisuProgram.init_server(
        cpp_code=cpp_code,
        load_isl_ast=True,
        load_tree=True,
        reuse_server=True,
    )


def _interchange_schedule(sample):
    schedule = Schedule(sample)
    schedule.add_optimizations(
        [tiramisu_actions.Interchange(params=[("x_temp", 0), ("x_temp", 1)])]
    )
    return schedule


def _illegal_schedule(sample):
    # Parallelizing the reduction loop of x_temp is illegal.
    schedule = Schedule(sample)
    schedule.add_optimizations(
        [tiramisu_actions.Parallelization(params=[("x_temp", 1)])]
    )
    return schedule


def test_serve_legality_matches_one_shot():
    sample = _server_program()

    legal_one_shot = _interchange_schedule(sample).is_legal()
    illegal_one_shot = _illegal_schedule(sample).is_legal()

    sample.server.start_serve(1)
    try:
        assert _interchange_schedule(sample).is_legal() is legal_one_shot
        assert _illegal_schedule(sample).is_legal() is illegal_one_shot
    finally:
        sample.server.stop_serve()


def test_serve_fast_legality_matches_default():
    sample = _server_program()
    sample.server.start_serve(1)
    try:
        assert _interchange_schedule(sample).is_legal(fast=True) is True
        assert _illegal_schedule(sample).is_legal(fast=True) is False
    finally:
        sample.server.stop_serve()


def test_serve_execute_returns_times_without_relegality():
    sample = _server_program()
    sample.server.start_serve(1)
    try:
        schedule = _interchange_schedule(sample)
        assert schedule.is_legal(fast=True) is True
        times = schedule.execute(min_runs=2, delete_files=False)
        assert len(times) == 2
        assert all(t > 0 for t in times)
        assert schedule.legality is True
    finally:
        sample.server.stop_serve()


def test_serve_codegen_then_run_obj():
    sample = _server_program()
    sample.server.start_serve(1)
    try:
        schedule = _interchange_schedule(sample)
        assert schedule.is_legal(fast=True) is True
        client = sample.server.serve_clients[0]
        tag = f"{sample.temp_files_identifier}_cand0"

        codegen_res = client.request("codegen", schedule_str=str(schedule), obj_tag=tag)
        assert codegen_res.success is True

        run_res = client.request("run_obj", min_runs=2, obj_tag=tag)
        assert run_res.success is True
        assert len(run_res.exec_times) == 2
    finally:
        sample.server.stop_serve()


def test_serve_run_obj_missing_tag_reports_failure():
    sample = _server_program()
    sample.server.start_serve(1)
    try:
        client = sample.server.serve_clients[0]
        res = client.request("run_obj", min_runs=1, obj_tag="does_not_exist")
        assert res.success is False
    finally:
        sample.server.stop_serve()


def test_serve_survives_failed_request():
    sample = _server_program()
    sample.server.start_serve(1)
    try:
        client = sample.server.serve_clients[0]
        with pytest.raises(ServerExecutionFailedError):
            client.request("no_such_operation")
        # the instance must remain usable after a failed request
        assert _interchange_schedule(sample).is_legal(fast=True) is True
    finally:
        sample.server.stop_serve()


def test_serve_annotations_matches_one_shot():
    sample = _server_program()
    one_shot = sample.server.get_annotations()
    sample.server.start_serve(1)
    try:
        assert sample.server.get_annotations() == one_shot
    finally:
        sample.server.stop_serve()
