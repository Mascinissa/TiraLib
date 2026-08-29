import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from tiralib.config import BaseConfig

if TYPE_CHECKING:
    from tiralib.tiramisu.schedule import Schedule
    from tiralib.tiramisu.tiramisu_program import TiramisuProgram

logger = logging.getLogger(__name__)


class ServerExecutionFailedError(Exception):
    """Exception raised when the server execution fails."""


def _expand(path_str: str) -> str:
    return os.path.expandvars(path_str)


def build_env() -> dict:
    """The environment the server compile/run commands used to assemble via
    `export` statements, as a dict usable with subprocess env=."""
    if not BaseConfig.base_config:
        raise ValueError("BaseConfig not initialized")
    env = os.environ.copy()
    for key, value in BaseConfig.base_config.env_vars.items():
        env[key] = _expand(str(value))
    libs = ":".join(_expand(p) for p in BaseConfig.base_config.dependencies.libs)
    env["LD_LIBRARY_PATH"] = libs + ":" + env.get("LD_LIBRARY_PATH", "")
    env["LIBRARY_PATH"] = libs + ":" + env.get("LIBRARY_PATH", "")
    env["CPATH"] = ":".join(
        _expand(p) for p in BaseConfig.base_config.dependencies.includes
    )
    return env


templateWithEverythinginUtils = """
#include <tiramisu/tiramisu.h>
#include <TiraLibCPP/actions.h>
#include <TiraLibCPP/utils.h>

using namespace tiramisu;

int main(int argc, char *argv[])
{{
    // check the number of arguemnts is 2 or 3
    assert(argc == 1 || argc == 2 || argc == 3 && "Invalid number of arguments");
    // get the operation to perform
    Operation operation = Operation::legality;

    if (argc >= 2)
    {{
        operation = get_operation_from_string(argv[1]);
    }}
    // get the schedule string if provided
    std::string schedule_str = "";
    if (argc == 3)
        schedule_str = argv[2];

    std::string function_name = "{name}";

    {body}

    schedule_str_to_result_str(function_name, schedule_str, operation, {buffers});
    return 0;
}}
"""  # noqa: E501


class ResultInterface:
    """Result interface for the function server."""

    def __init__(self, result_str: bytes) -> None:
        """Initialize the result interface.

        Args:
            result_str (bytes): The result string.
        """
        decoded = result_str.decode("utf-8")
        decoded = decoded.strip().replace("\n", "\\n")

        self.halide_ir = None
        # extract halide ir and the result dict
        if "Generated Halide" in decoded:
            regex = r"Generated Halide IR:([\w\W\s]*)(?=\{\"name)(.*)"
            match = re.search(regex, decoded, re.MULTILINE | re.DOTALL)
            if match is None:
                raise ValueError(f"Could not parse the result string: {decoded}")
            self.halide_ir = match.group(1)
            logger.debug(self.halide_ir.replace("\\n", "\n"))
            decoded = match.group(2)
        result_dict = json.loads(decoded)

        self.name: str = result_dict["name"]
        self.legality: bool = result_dict["legality"] == 1
        self.isl_ast: str = result_dict["isl_ast"]
        self.success: bool = bool(result_dict["success"])

        # convert exec_times to list of floats
        self.exec_times = (
            [float(x) for x in result_dict["exec_times"].split()]
            if result_dict["exec_times"]
            else []
        )
        self.additional_info = (
            result_dict["additional_info"] if "additional_info" in result_dict else None
        )

    def __str__(self) -> str:
        """Return a string representation of the object."""
        isl_ast = self.isl_ast.replace("\n", ",")
        return f"ResultInterface(name={self.name},legality={self.legality},isl_ast={isl_ast},exec_times={self.exec_times},success={self.success})"  # noqa: E501

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return self.__str__()


class FunctionServer:
    """Function server class."""

    def __init__(self, tiramisu_program: "TiramisuProgram", reuse_server: bool = False):
        if not BaseConfig.base_config:
            raise ValueError("BaseConfig not initialized")

        if not tiramisu_program.cpp_code:
            raise ValueError("Tiramisu program not initialized")

        self.tiramisu_program = tiramisu_program

        server_path_cpp = (
            Path(BaseConfig.base_config.workspace)
            / f"{tiramisu_program.temp_files_identifier}_server.cpp"
        )

        server_path = (
            Path(BaseConfig.base_config.workspace)
            / f"{tiramisu_program.temp_files_identifier}_server"
        )

        if reuse_server and server_path.exists():
            logger.info("Server code already exists. Skipping generation")
            return

        # Generate the server code
        server_code = FunctionServer._generate_server_code_from_program(
            tiramisu_program
        )

        # Write the server code to a file
        server_path_cpp.write_text(server_code)

        # Write the wrapper code to a file
        wrapper_path = (
            Path(BaseConfig.base_config.workspace)
            / f"{tiramisu_program.temp_files_identifier}_wrapper.cpp"
        )
        wrapper_path.write_text(tiramisu_program.wrappers["cpp"])

        # Write the wrapper header to a file
        wrapper_header_path = (
            Path(BaseConfig.base_config.workspace)
            / f"{tiramisu_program.temp_files_identifier}_wrapper.h"
        )

        wrapper_header_path.write_text(tiramisu_program.wrappers["h"])

        # compile the server code
        self._compile_server_code()

    @classmethod
    def _generate_server_code_from_program(cls, tiramisu_program: "TiramisuProgram"):
        # fill the template
        function_str = templateWithEverythinginUtils.format(
            name=tiramisu_program.temp_files_identifier,
            body=tiramisu_program.body,
            buffers="{&" + ", &".join(tiramisu_program.IO_buffer_names) + "}",
        )
        return function_str

    def _compile_server_code(self):
        """Compile the server code."""
        if not BaseConfig.base_config:
            raise ValueError("BaseConfig not initialized")

        env = build_env()
        ws = BaseConfig.base_config.workspace
        name = self.tiramisu_program.temp_files_identifier

        use_sqlite = "-lsqlite3" if BaseConfig.base_config.tiralib_cpp.use_sqlite else ""
        compile_command = (
            f"$CXX -fvisibility-inlines-hidden -ftree-vectorize -fstack-protector-strong "
            f"-fno-plt -O3 -ffunction-sections -pipe -ldl -g -fno-rtti -lpthread -std=c++17 "
            f"-MD -MT {name}.cpp.o -MF {name}.cpp.o.d -o {name}.cpp.o -c {name}_server.cpp && "
            f"$CXX -fvisibility-inlines-hidden -ftree-vectorize -fstack-protector-strong "
            f"-fno-plt -O3 -ffunction-sections -pipe -ldl -g -fno-rtti -lpthread "
            f"{name}.cpp.o -o {name}_server "
            f"-ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl -lTiraLibCPP {use_sqlite} -lz"
        )

        # run the command and retrieve the execution status
        try:
            subprocess.check_output(
                compile_command, shell=True, cwd=ws, env=env, stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Error while compiling server code: {e}")
            logger.error(e.output)
            raise e

    def run(
        self,
        operation: Literal["execution", "legality"] = "legality",
        schedule: "Schedule | None" = None,
        min_runs: int = 1,
        max_runs: int | None = None,
        time_budget: float | None = None,
        delete_files: bool = False,
    ):
        """Run the server code."""
        if not BaseConfig.base_config:
            raise ValueError("BaseConfig not initialized")
        assert operation in [
            "execution",
            "legality",
        ], (
            f"Invalid operation {operation}. Valid operations are: execution, legality, annotations"
        )  # noqa: E501

        legality_result = None
        server_operation = operation
        if operation == "execution" and schedule is not None:
            # The execution replay serializes subset unrolling as plain U(...),
            # which Tiramisu's transformed-schedule legality path cannot
            # represent; legality must come from the UCheck serialization.
            # When the schedule's legality is already known (an explicit
            # is_legal() ran that exact check), re-running it here would be
            # pure duplicated work — trust it and go straight to the replay.
            if schedule.legality is True:
                server_operation = "execution_no_check"
            else:
                legality_result = self.run(
                    operation="legality",
                    schedule=schedule,
                    min_runs=min_runs,
                    max_runs=max_runs,
                    time_budget=time_budget,
                    delete_files=False,
                )
                if not legality_result.legality:
                    if delete_files:
                        self.delete_temporary_files()
                    return legality_result
                server_operation = "execution_no_check"

        if operation == "legality" and schedule is not None:
            schedule_str = schedule.get_legality_str()
        else:
            schedule_str = str(schedule or "")

        env = build_env()
        ws = BaseConfig.base_config.workspace
        command = (
            f'MIN_RUNS={min_runs} MAX_RUNS={max_runs if max_runs else "inf"} '
            f'TIME_BUDGET={time_budget if time_budget else "-1"} '
            f'./{self.tiramisu_program.temp_files_identifier}_server '
            f'{server_operation} "{schedule_str}"'
        )

        # run the command and retrieve the execution status
        try:
            output = subprocess.check_output(command, shell=True, cwd=ws, env=env)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error while running server code: {e}")
            logger.error(e.output)
            logger.error(e.stderr)
            raise e
        if delete_files:
            self.delete_temporary_files()
        result = ResultInterface(output)
        if legality_result is not None:
            result.legality = legality_result.legality
        elif operation == "execution" and schedule is not None and schedule.legality is True:
            # execution_no_check trusted the caller's legality; reflect it back.
            result.legality = True
        return result

    def get_annotations(self):
        """Run the server code to get the annotations."""
        if not BaseConfig.base_config:
            raise ValueError("BaseConfig not initialized")
        env = build_env()
        ws = BaseConfig.base_config.workspace
        command = f"./{self.tiramisu_program.temp_files_identifier}_server annotations"

        # run the command and retrieve the execution status
        try:
            output = subprocess.check_output(command, shell=True, cwd=ws, env=env)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error while running server code: {e}")
            logger.error(e.output)
            logger.error(e.stderr)
            raise e

        return output.decode("utf-8").strip()

    def delete_temporary_files(self):
        """Delete files temporary files"""
        assert BaseConfig.base_config, "BaseConfig not initialized"
        subprocess.run(
            [
                # cd to the workspace and clean generated files
                f"cd {BaseConfig.base_config.workspace} && rm {self.tiramisu_program.temp_files_identifier}*{{.o,.cpp,.h,.so,.d}}"  # noqa: E501
            ],
            capture_output=True,
            text=True,
            shell=True,
            check=False,
            executable="/bin/bash",
        )
