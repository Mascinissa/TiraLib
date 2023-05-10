import re
from pathlib import Path

from athena.utils.config import BaseConfig


class TiramisuProgram:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.annotations = None
        self.comps = None
        self.name = None
        self.schedules_legality = {}
        self.schedules_solver = {}
        self.original_str = None
        self.wrappers = None
        self.initial_execution_times = {}
        self.current_machine_initial_execution_time = None
        if file_path:
            self.load_code_lines()

    # Since there is no factory constructors in python, I am creating this class method to replace the factory pattern
    @classmethod
    def from_dict(
        cls, name: str, data: dict, original_str: str = None, wrappers: dict = None
    ):
        # Initiate an instante of the TiramisuProgram class
        tiramisu_prog = cls(None)
        tiramisu_prog.name = name
        tiramisu_prog.annotations = data["program_annotation"]
        if tiramisu_prog.annotations:
            tiramisu_prog.comps = list(tiramisu_prog.annotations["computations"].keys())
            tiramisu_prog.schedules_legality = data["schedules_legality"]
            tiramisu_prog.schedules_solver = data["schedules_solver"]

            # Initialize the initial_execution_times attribute and the current_machine_initial_execution_time attribute
            tiramisu_prog.initial_execution_times = data["initial_execution_times"]
            # if cfg.Config.config.tiramisu.hpc_name in data["initial_execution_times"]:
            #     tiramisu_prog.current_machine_initial_execution_time = min(data[
            #         "initial_execution_times"][cfg.Config.config.tiramisu.hpc_name])

        tiramisu_prog.load_code_lines(original_str)

        if wrappers:
            tiramisu_prog.wrappers = wrappers

        # If the current_machine_initial_execution_time attribute is not found in the data, compute itcio
        # if not tiramisu_prog.current_machine_initial_execution_time:
        #     tmp_exec_times = CompilingModule.CompilingService.get_cpu_exec_times(
        #         tiramisu_program=tiramisu_prog, optims_list=[])
        #     # Store the minimum execution time in the initial_execution_time attribute
        #     tiramisu_prog.current_machine_initial_execution_time = min(
        #         tmp_exec_times)

        #     tiramisu_prog.initial_execution_times[
        #         cfg.Config.config.tiramisu.hpc_name] = tmp_exec_times

        # After taking the neccessary fields return the instance
        return tiramisu_prog

    @classmethod
    def from_file(cls, file_path: str, wrapper_cpp_path: str, wrapper_header_path: str):
        # Initiate an instante of the TiramisuProgram class
        tiramisu_prog = cls(file_path)

        # load the wrapper code
        with open(wrapper_cpp_path, "r") as f:
            wrapper_cpp = f.read()
        with open(wrapper_header_path, "r") as f:
            wrapper_header = f.read()

        tiramisu_prog.wrappers = {"cpp": wrapper_cpp, "h": wrapper_header}

        # After taking the neccessary fields return the instance
        return tiramisu_prog

    def load_code_lines(self, original_str: str = None):
        """
        This function loads the file code , it is necessary to generate legality check code and annotations
        """
        if self.name:
            # if self.name is None the program doesn't exist in the offline dataset but built from compiling
            # if self.name has a value than it is fetched from the dataset, we need the full path to read
            # the lines of the real function to execute legality code
            func_name = self.name
            file_name = func_name + "_generator.cpp"
            file_path = (
                BaseConfig.base_config.dataset.cpps_path + func_name + "/" + file_name
            )
            self.file_path = file_path
        else:
            file_path = self.file_path

        if original_str:
            self.original_str = original_str
        else:
            with open(file_path, "r") as f:
                self.original_str = f.read()

        self.func_folder = (
            "/".join(Path(file_path).parts[:-1])
            if len(Path(file_path).parts) > 1
            else "."
        ) + "/"
        self.body = re.findall(
            r"(tiramisu::init(?s:.)+)tiramisu::codegen", self.original_str
        )[0]
        self.name = re.findall(r"tiramisu::init\(\"(\w+)\"\);", self.original_str)[0]
        # Remove the wrapper include from the original string
        self.wrapper_str = f'#include "{self.name}_wrapper.h"'
        self.original_str = self.original_str.replace(
            self.wrapper_str, f"// {self.wrapper_str}"
        )
        self.comps = re.findall(r"computation (\w+)\(", self.original_str)
        self.code_gen_line = re.findall(r"tiramisu::codegen\({.+;", self.original_str)[
            0
        ]
        # buffers_vect = re.findall(r'{(.+)}', self.code_gen_line)[0]
        # self.IO_buffer_names = re.findall(r'\w+', buffers_vect)
        # self.buffer_sizes = []
        # for buf_name in self.IO_buffer_names:
        #     sizes_vect = re.findall(r'buffer ' + buf_name + '.*{(.*)}',
        #                             self.original_str)[0]
        #     self.buffer_sizes.append(re.findall(r'\d+', sizes_vect))

    def __str__(self) -> str:
        return f"TiramisuProgram(name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()
