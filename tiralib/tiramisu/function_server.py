import itertools
import json
import logging
import os
import re
import subprocess
import threading
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


# The server binary is pure glue (function construction + request dispatch); the
# schedules it times run from Halide-generated objects whose compilation is
# independent, so optimizing the glue wastes seconds per program. -O0/-g0 halves
# the per-program server compile; the shared precompiled header (identical
# includes for every program) removes most of the rest. Override with
# TIRALIB_SERVER_CXXFLAGS if needed.
SERVER_CXXFLAGS = os.environ.get(
    "TIRALIB_SERVER_CXXFLAGS", "-O0 -g0 -fno-rtti -std=c++17 -pipe"
)
PCH_HEADER_NAME = "tiralib_pch.h"
PCH_CONTENT = (
    "#include <tiramisu/tiramisu.h>\n"
    "#include <TiraLibCPP/actions.h>\n"
    "#include <TiraLibCPP/utils.h>\n"
)


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


# Bumped whenever the generated server code changes shape; binaries report it
# via `./<name>_server template_version` so reuse_server can detect stale ones.
SERVER_TEMPLATE_VERSION = "tiralib-server-2"

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

    if (argc >= 2 && std::string(argv[1]) == "template_version")
    {{
        std::cout << "{template_version}" << std::endl;
        return 0;
    }}

    std::string function_name = "{name}";

    {body}

    // persistent fork-server mode: one request per stdin line (see run_server_loop)
    if (argc >= 2 && std::string(argv[1]) == "serve")
    {{
        return run_server_loop(function_name, {buffers});
    }}

    if (argc >= 2)
    {{
        operation = get_operation_from_string(argv[1]);
    }}
    // get the schedule string if provided
    std::string schedule_str = "";
    if (argc == 3)
        schedule_str = argv[2];

    schedule_str_to_result_str(function_name, schedule_str, operation, {buffers});
    return 0;
}}
"""  # noqa: E501

SERVE_READY = "###TIRALIB_SERVE_READY###"
SERVE_DONE = "###TIRALIB_SERVE_DONE###"
SERVE_FAIL = "###TIRALIB_SERVE_FAIL"


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


class ServeClient:
    """Client for one persistent fork-server instance (`./<ident>_server serve`).

    The server builds the tiramisu function and runs dependency analysis once;
    each request is then handled by a forked child (fresh schedule state, crash
    isolation) without re-paying process spawn, dylib load, function build or
    dependency analysis. Requests on one instance are strictly serial (guarded
    by a lock); use several instances for parallelism (e.g. concurrent legality
    checks from a thread pool).
    """

    def __init__(
        self,
        server_binary: str,
        workspace: str,
        env: dict,
        stderr_path: str | None = None,
    ):
        self.server_binary = server_binary
        self.workspace = workspace
        # serializes requests on this instance so several threads can share a
        # pool of instances safely (one in-flight request per instance)
        self.lock = threading.Lock()
        self._stderr_f = open(stderr_path, "ab") if stderr_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            [f"./{server_binary}", "serve"],
            cwd=workspace,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_f,
        )
        self._read_until(SERVE_READY)

    def _read_until(self, sentinel: str) -> bytes:
        collected = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise ServerExecutionFailedError(
                    f"serve process {self.server_binary} died "
                    f"(exit={self.proc.poll()})"
                )
            text = line.decode("utf-8", errors="replace").strip()
            if text == sentinel:
                return b"".join(collected)
            if text.startswith(SERVE_FAIL):
                raise ServerExecutionFailedError(
                    f"serve child failed: {text} (partial output: "
                    f"{b''.join(collected)[-300:]!r})"
                )
            collected.append(line)

    def request(
        self,
        op: str,
        schedule_str: str = "",
        min_runs: int = 1,
        max_runs: int | None = None,
        time_budget: float | None = None,
        obj_tag: str = "",
    ) -> "ResultInterface":
        if self.proc.poll() is not None:
            raise ServerExecutionFailedError(
                f"serve process {self.server_binary} is not running"
            )
        line = "\t".join(
            [
                op,
                str(min_runs),
                str(max_runs) if max_runs else "inf",
                str(time_budget) if time_budget else "-1",
                obj_tag,
                schedule_str,
            ]
        )
        with self.lock:
            self.proc.stdin.write((line + "\n").encode("utf-8"))
            self.proc.stdin.flush()
            output = self._read_until(SERVE_DONE)
        return ResultInterface(output)

    def annotations(self) -> str:
        if self.proc.poll() is not None:
            raise ServerExecutionFailedError(
                f"serve process {self.server_binary} is not running"
            )
        with self.lock:
            self.proc.stdin.write(b"annotations\t1\tinf\t-1\t\t\n")
            self.proc.stdin.flush()
            return self._read_until(SERVE_DONE).decode("utf-8").strip()

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write(b"exit\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        finally:
            if self._stderr_f is not subprocess.DEVNULL:
                try:
                    self._stderr_f.close()
                except Exception:
                    pass


class FunctionServer:
    """Function server class."""

    def __init__(self, tiramisu_program: "TiramisuProgram", reuse_server: bool = False):
        if not BaseConfig.base_config:
            raise ValueError("BaseConfig not initialized")

        if not tiramisu_program.cpp_code:
            raise ValueError("Tiramisu program not initialized")

        self.tiramisu_program = tiramisu_program
        self._serve_clients: list[ServeClient] = []
        self._rr = itertools.count()

        server_path_cpp = (
            Path(BaseConfig.base_config.workspace)
            / f"{tiramisu_program.temp_files_identifier}_server.cpp"
        )

        server_path = (
            Path(BaseConfig.base_config.workspace)
            / f"{tiramisu_program.temp_files_identifier}_server"
        )

        if reuse_server and server_path.exists():
            if self._binary_template_version(server_path) == SERVER_TEMPLATE_VERSION:
                logger.info("Server code already exists. Skipping generation")
                return
            logger.info(
                "Existing server binary was generated from an older template; "
                "regenerating"
            )

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
            template_version=SERVER_TEMPLATE_VERSION,
        )
        return function_str

    @staticmethod
    def _binary_template_version(server_path) -> str | None:
        """Ask an existing server binary which template generated it. Binaries
        from before the version handshake abort on the unknown argument and
        report None, which callers treat as stale."""
        try:
            out = subprocess.check_output(
                [str(server_path), "template_version"],
                env=build_env(),
                cwd=str(server_path.parent),
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
            return out.decode("utf-8").strip()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Persistent serve mode
    # ------------------------------------------------------------------
    def start_serve(self, n_instances: int = 1, stderr_path: str | None = None):
        """Start (or grow to) n_instances persistent serve processes."""
        assert BaseConfig.base_config
        env = build_env()
        ws = str(BaseConfig.base_config.workspace)
        name = f"{self.tiramisu_program.temp_files_identifier}_server"
        while len(self._serve_clients) < n_instances:
            self._serve_clients.append(
                ServeClient(name, ws, env, stderr_path=stderr_path)
            )
        return self._serve_clients

    @property
    def serve_clients(self) -> list["ServeClient"]:
        return self._serve_clients

    def stop_serve(self):
        for c in self._serve_clients:
            c.close()
        self._serve_clients = []

    @classmethod
    def _pch_stamp(cls, env: dict) -> str:
        """Fingerprint of everything the precompiled header depends on: compiler,
        flags, and the modification times of the directly-included headers
        (resolved through CPATH). Rebuilding on any change keeps a stale .gch
        from breaking or silently mismatching after a TiraLibCPP/tiramisu
        update. (Deep-include edits inside tiramisu are not tracked; wipe the
        workspace or touch a top-level header after those.)"""
        try:
            cxx_id = subprocess.check_output(
                "$CXX --version", shell=True, env=env, text=True
            ).splitlines()[0]
        except Exception:
            cxx_id = "unknown"
        mtimes = []
        for rel in ("tiramisu/tiramisu.h", "TiraLibCPP/actions.h", "TiraLibCPP/utils.h"):
            for inc_dir in env.get("CPATH", "").split(":"):
                cand = Path(inc_dir) / rel
                if inc_dir and cand.exists():
                    mtimes.append(f"{rel}:{cand.stat().st_mtime_ns}")
                    break
        return json.dumps({"cxx": cxx_id, "flags": SERVER_CXXFLAGS, "headers": mtimes})

    @classmethod
    def _ensure_pch(cls, env: dict) -> str | None:
        """Build the shared precompiled header once per workspace (every server
        .cpp shares the exact same includes), rebuilding when the compiler,
        flags or headers changed. Returns the header name for -include, or
        None if the PCH could not be built (compilation then proceeds without
        it, just slower)."""
        assert BaseConfig.base_config
        ws = Path(BaseConfig.base_config.workspace)
        pch_h = ws / PCH_HEADER_NAME
        pch_gch = ws / (PCH_HEADER_NAME + ".gch")
        stamp_file = ws / (PCH_HEADER_NAME + ".stamp")
        stamp = cls._pch_stamp(env)
        if pch_gch.exists() and stamp_file.exists() and stamp_file.read_text() == stamp:
            return str(pch_h.name)
        try:
            pch_h.write_text(PCH_CONTENT)
            cmd = f"$CXX {SERVER_CXXFLAGS} -x c++-header {pch_h.name} -o {pch_gch.name}"
            subprocess.check_output(
                cmd, shell=True, cwd=ws, env=env, stderr=subprocess.STDOUT
            )
            stamp_file.write_text(stamp)
            return str(pch_h.name)
        except subprocess.CalledProcessError as e:
            logger.warning(f"PCH build failed (continuing without): {e.output}")
            pch_gch.unlink(missing_ok=True)
            stamp_file.unlink(missing_ok=True)
            return None

    def _compile_server_code(self):
        """Compile the server code."""
        if not BaseConfig.base_config:
            raise ValueError("BaseConfig not initialized")

        env = build_env()
        ws = BaseConfig.base_config.workspace
        name = self.tiramisu_program.temp_files_identifier

        pch = FunctionServer._ensure_pch(env)
        include_pch = f"-include {pch} " if pch else ""

        use_sqlite = "-lsqlite3" if BaseConfig.base_config.tiralib_cpp.use_sqlite else ""
        compile_command = (
            f"$CXX {SERVER_CXXFLAGS} {include_pch}-o {name}.cpp.o -c {name}_server.cpp && "
            f"$CXX {SERVER_CXXFLAGS} {name}.cpp.o -o {name}_server "
            f"-ldl -lpthread -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl "
            f"-lTiraLibCPP {use_sqlite} -lz"
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
            "legality_noast",
        ], (
            f"Invalid operation {operation}. Valid operations are: execution, legality, legality_noast, annotations"
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

        if operation in ("legality", "legality_noast") and schedule is not None:
            schedule_str = schedule.get_legality_str()
        else:
            schedule_str = str(schedule or "")

        # Serve mode: route through a persistent instance (no process spawn,
        # no dylib reload, no function rebuild, no dependency re-analysis).
        # Prefer an idle instance so threaded callers get real parallelism.
        if self._serve_clients:
            client = next(
                (c for c in self._serve_clients if not c.lock.locked()),
                self._serve_clients[next(self._rr) % len(self._serve_clients)],
            )
            result = client.request(
                server_operation,
                schedule_str=schedule_str,
                min_runs=min_runs,
                max_runs=max_runs,
                time_budget=time_budget,
            )
            if delete_files:
                self.delete_temporary_files()
            if legality_result is not None:
                result.legality = legality_result.legality
            elif (
                operation == "execution"
                and schedule is not None
                and schedule.legality is True
            ):
                result.legality = True
            return result

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
        if self._serve_clients:
            return self._serve_clients[0].annotations()
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
